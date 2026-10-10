"""Nodo Supervisor: router inteligente y controlador de flujo.

Decide, en cada vuelta, quién interviene ahora (investigador, analista) o si ya
es momento de finalizar. Valida el trabajo con una rúbrica explícita antes de
permitir el cierre y tiene dos frenos DUROS en código (no dependen del LLM):

- `MAX_PASOS`: tope de decisiones del supervisor (criterio de parada).
- `MAX_LLAMADAS_POR_AGENTE`: cuántas veces se puede mandar a refinar a un mismo
  especialista, para evitar bucles de correcciones eternas.
"""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from state import OrchestratorState

MAX_PASOS = 6
MAX_LLAMADAS_POR_AGENTE = 2
MAX_CARACTERES_CONTRIBUCION = 1500  # lo que ve el supervisor de cada aporte

SUPERVISOR_PROMPT = """\
Sos el SUPERVISOR de un equipo de dos especialistas. Dada la consulta del usuario
y lo que el equipo ya aportó, decidí quién debe intervenir ahora o si es momento
de finalizar.

ESPECIALISTAS
- investigador: busca datos en la base de conocimiento (reseñas con calificación,
  fichas con comisiones y costos). Solo recupera datos; no calcula.
- analista: calcula y analiza (promedios, sentimiento, costos acumulados). No
  tiene acceso a la base: solo trabaja con los datos que le pasás.

CÓMO DECIDIR
1. Si todavía faltan datos de la base -> investigador.
2. Si ya hay datos pero faltan cálculos o análisis -> analista, y en `instruccion`
   copiá los datos necesarios, completos, para que pueda trabajar sin más contexto.
3. Si un aporte es incompleto, tiene errores o no respeta la rúbrica, podés
   devolverlo al mismo especialista con una instrucción concreta de qué corregir
   (refinamiento). Hacelo solo si de verdad hace falta.
4. Si la rúbrica se cumple, elegí FINISH.

RÚBRICA DE VALIDACIÓN (completá los tres campos con honestidad)
- datos_con_fuente: los datos que se van a usar vienen de la base e indican su
  fuente o id.
- calculos_hechos_por_analista: todo número derivado (promedios, totales,
  porcentajes) fue calculado por el analista con sus herramientas, no inventado.
- responde_toda_la_consulta: cada parte de la pregunta del usuario está cubierta.
Solo podés elegir FINISH si los tres son verdaderos, o si no hay forma de mejorar
el resultado con las herramientas disponibles (en ese caso, explicalo en `razon`).

`instruccion` debe ser autocontenida: el especialista NO ve la conversación, solo
esa instrucción. Dejala vacía si elegís FINISH."""


class Decision(BaseModel):
    """Salida estructurada del supervisor. Los campos de la rúbrica van primero a
    propósito: el modelo evalúa antes de decidir."""

    razon: str = Field(description="Breve justificación de la decisión (1-2 frases).")
    datos_con_fuente: bool
    calculos_hechos_por_analista: bool
    responde_toda_la_consulta: bool
    siguiente: Literal["investigador", "analista", "FINISH"]
    instruccion: str = Field(
        default="", description="Instrucción autocontenida para el especialista elegido."
    )


def _pregunta_del_usuario(state: OrchestratorState) -> str:
    for mensaje in state["messages"]:
        if isinstance(mensaje, HumanMessage):
            return str(mensaje.content)
    return ""


def _resumen_del_equipo(state: OrchestratorState) -> str:
    contribuciones = state.get("contribuciones", [])
    if not contribuciones:
        return "(todavía nadie aportó nada)"
    bloques = []
    for c in contribuciones:
        texto = c["resultado"]
        if len(texto) > MAX_CARACTERES_CONTRIBUCION:
            texto = texto[:MAX_CARACTERES_CONTRIBUCION] + "... [recortado]"
        bloques.append(
            f"[paso {c['paso']}] {c['agente']} (herramientas: "
            f"{', '.join(c['herramientas']) or 'ninguna'}):\n{texto}"
        )
    return "\n\n".join(bloques)


def crear_supervisor(llm: BaseChatModel):
    decisor = llm.with_structured_output(Decision)

    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("pasos", 0) + 1
        contribuciones = state.get("contribuciones", [])
        llamadas = {
            a: sum(c["agente"] == a for c in contribuciones)
            for a in ("investigador", "analista")
        }

        # Freno duro 1: tope de pasos. No se consulta al LLM.
        if paso > MAX_PASOS:
            decision = Decision(
                razon=f"Se alcanzó el tope de {MAX_PASOS} decisiones: se cierra con lo disponible.",
                datos_con_fuente=False,
                calculos_hechos_por_analista=False,
                responde_toda_la_consulta=False,
                siguiente="FINISH",
            )
        else:
            decision = await decisor.ainvoke(
                [
                    SystemMessage(content=SUPERVISOR_PROMPT),
                    HumanMessage(
                        content=(
                            f"CONSULTA DEL USUARIO:\n{_pregunta_del_usuario(state)}\n\n"
                            f"APORTES DEL EQUIPO HASTA AHORA:\n{_resumen_del_equipo(state)}\n\n"
                            f"Decisión número {paso} de un máximo de {MAX_PASOS}. "
                            f"Veces que intervino cada especialista: {llamadas}."
                        )
                    ),
                ]
            )

        siguiente = decision.siguiente
        razon = decision.razon
        instruccion = decision.instruccion.strip()

        # Freno duro 2: límite de refinamientos por especialista.
        if siguiente != "FINISH" and llamadas[siguiente] >= MAX_LLAMADAS_POR_AGENTE:
            razon += f" [forzado a FINISH: '{siguiente}' ya intervino {MAX_LLAMADAS_POR_AGENTE} veces]"
            siguiente = "FINISH"
        # Una instrucción vacía dejaría al especialista sin saber qué hacer.
        if siguiente != "FINISH" and not instruccion:
            instruccion = _pregunta_del_usuario(state)

        registro = {
            "paso": paso,
            "siguiente": siguiente,
            "razon": razon,
            "instruccion": instruccion if siguiente != "FINISH" else "",
            "evaluacion": {
                "datos_con_fuente": decision.datos_con_fuente,
                "calculos_hechos_por_analista": decision.calculos_hechos_por_analista,
                "responde_toda_la_consulta": decision.responde_toda_la_consulta,
            },
        }
        return {
            "next_agent": siguiente,
            "instruccion": registro["instruccion"],
            "pasos": paso,
            "decisiones": [registro],
        }

    return supervisor


def route_supervisor(
    state: OrchestratorState,
) -> Literal["investigador", "analista", "sintesis"]:
    """Arista condicional: traduce la decisión del supervisor al nombre del
    próximo nodo. FINISH lleva a la síntesis final."""
    siguiente = state["next_agent"]
    if siguiente == "investigador":
        return "investigador"
    if siguiente == "analista":
        return "analista"
    return "sintesis"
