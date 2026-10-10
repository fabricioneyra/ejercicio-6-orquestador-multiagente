"""Estado compartido del orquestador.

Es la "pizarra" común: ningún agente ve la conversación de otro; todo lo que un
especialista aporta queda registrado acá, firmado con su nombre. Eso permite
rastrear QUÉ agente aportó QUÉ información y evita perder contexto.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class Contribucion(TypedDict):
    """Un aporte firmado de un especialista."""

    paso: int  # número de paso del supervisor que originó este trabajo
    agente: str  # "investigador" | "analista"
    instruccion: str  # lo que el supervisor le pidió
    resultado: str  # lo que el agente respondió
    herramientas: list[str]  # herramientas que usó (trazabilidad)


class Evaluacion(TypedDict):
    """Rúbrica con la que el supervisor valida el trabajo antes de finalizar."""

    datos_con_fuente: bool
    calculos_hechos_por_analista: bool
    responde_toda_la_consulta: bool


class DecisionRegistrada(TypedDict):
    """Una decisión del supervisor (queda en el estado para la traza)."""

    paso: int
    siguiente: str
    razon: str
    instruccion: str
    evaluacion: Evaluacion


class OrchestratorState(MessagesState):
    """Hereda `messages` de MessagesState (pregunta del usuario, aportes
    firmados y respuesta final) y suma campos propios.

    `contribuciones` y `decisiones` usan `operator.add` como reducer: cada nodo
    devuelve una lista con SUS elementos nuevos y LangGraph los concatena, en
    vez de pisar lo que escribieron otros nodos.
    """

    next_agent: str  # "investigador" | "analista" | "FINISH" (lo fija el supervisor)
    instruccion: str  # instrucción puntual para el próximo especialista
    pasos: int  # cuántas veces decidió el supervisor (contador anti-bucle)
    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[DecisionRegistrada], operator.add]
    task_completed: bool
