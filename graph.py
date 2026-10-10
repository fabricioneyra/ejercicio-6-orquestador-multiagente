"""Grafo del orquestador multi-agente (topología jerárquica con supervisor).

    START -> supervisor --(investigador)--> investigador --\\
                 ^  \\---(analista)-------> analista --------+--> supervisor (vuelve a decidir)
                 |   \\--(FINISH)--> sintesis -> END
                 +------------------------------------------/
"""

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import create_analyst_agent
from agents.research_agent import create_research_agent
from state import Contribucion, OrchestratorState
from supervisor import (
    _pregunta_del_usuario,
    _resumen_del_equipo,
    crear_supervisor,
    route_supervisor,
)

DEFAULT_MODEL = "gpt-4o-mini"
# Techo de pasos del grafo principal por ejecución: cada vuelta del ciclo
# (supervisor + especialista) consume 2, y el supervisor ya se limita a MAX_PASOS.
RECURSION_LIMIT = 25
# Techo para el ciclo interno de herramientas de cada especialista.
SUBAGENT_RECURSION_LIMIT = 12

SINTESIS_PROMPT = (
    "Sos el redactor final de un equipo de análisis. Con la consulta del usuario y los "
    "aportes del equipo, escribí la respuesta final en español: clara, breve y "
    "organizada. Usá SOLO los datos y cálculos de los aportes (no inventes ni "
    "recalcules). Citá la fuente de los datos cuando la tengan. Si algo no se pudo "
    "resolver, decilo con claridad."
)


def create_llm() -> BaseChatModel:
    """LLM de OpenAI. Lee OPENAI_API_KEY del entorno (.env)."""
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), temperature=0)


def _contexto_para(agente: str, state: OrchestratorState) -> str:
    """Contexto MÍNIMO que necesita cada especialista (evita la contaminación de
    contexto): el analista recibe los datos que trajo el investigador; el
    investigador no recibe nada más que su instrucción."""
    if agente != "analista":
        return ""
    datos = [c["resultado"] for c in state.get("contribuciones", []) if c["agente"] == "investigador"]
    return "\n\n".join(datos)


def _crear_nodo_especialista(nombre: str, agente: CompiledStateGraph):
    async def nodo(state: OrchestratorState) -> dict[str, Any]:
        instruccion = state["instruccion"]
        contexto = _contexto_para(nombre, state)
        mensaje = instruccion
        if contexto:
            mensaje += f"\n\nDATOS DISPONIBLES (aportados por el investigador):\n{contexto}"

        herramientas: list[str] = []
        try:
            # El especialista recibe SOLO este mensaje, no el historial del grafo.
            resultado = await agente.ainvoke(
                {"messages": [HumanMessage(content=mensaje)]},
                {"recursion_limit": SUBAGENT_RECURSION_LIMIT},
            )
            mensajes = resultado["messages"]
            texto = str(mensajes[-1].content)
            herramientas = [
                tc["name"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls
            ]
        except GraphRecursionError:
            texto = "ERROR: el especialista no logró terminar dentro del límite de pasos."

        contribucion: Contribucion = {
            "paso": state["pasos"],
            "agente": nombre,
            "instruccion": instruccion,
            "resultado": texto,
            "herramientas": herramientas,
        }
        return {
            "contribuciones": [contribucion],
            "messages": [AIMessage(content=texto, name=nombre)],
        }

    return nodo


def _crear_nodo_sintesis(llm: BaseChatModel):
    async def sintesis(state: OrchestratorState) -> dict[str, Any]:
        respuesta = await llm.ainvoke(
            [
                SystemMessage(content=SINTESIS_PROMPT),
                HumanMessage(
                    content=(
                        f"CONSULTA DEL USUARIO:\n{_pregunta_del_usuario(state)}\n\n"
                        f"APORTES DEL EQUIPO:\n{_resumen_del_equipo(state)}"
                    )
                ),
            ]
        )
        return {
            "messages": [AIMessage(content=respuesta.content, name="sintesis")],
            "task_completed": True,
        }

    return sintesis


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """Arma y compila el orquestador. `llm` se inyecta para poder testear con un
    modelo falso."""
    graph = StateGraph(OrchestratorState)
    graph.add_node("supervisor", crear_supervisor(llm))
    graph.add_node("investigador", _crear_nodo_especialista("investigador", create_research_agent(llm)))
    graph.add_node("analista", _crear_nodo_especialista("analista", create_analyst_agent(llm)))
    graph.add_node("sintesis", _crear_nodo_sintesis(llm))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"investigador": "investigador", "analista": "analista", "sintesis": "sintesis"},
    )
    graph.add_edge("investigador", "supervisor")  # los especialistas siempre reportan al supervisor
    graph.add_edge("analista", "supervisor")
    graph.add_edge("sintesis", END)
    return graph.compile()


def estado_inicial(pregunta: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=pregunta)],
        "next_agent": "",
        "instruccion": "",
        "pasos": 0,
        "contribuciones": [],
        "decisiones": [],
        "task_completed": False,
    }
