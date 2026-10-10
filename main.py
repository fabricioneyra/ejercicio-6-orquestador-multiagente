"""Demo del orquestador: muestra cómo el supervisor delega y cierra.

Uso:
    python main.py                       # consulta de ejemplo; guarda la traza
    python main.py --pregunta "..."      # tu propia consulta
    python main.py --graph               # imprime el diagrama Mermaid del grafo
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from graph import RECURSION_LIMIT, build_graph, create_llm, estado_inicial

TRACE_JSON = Path("traza_orquestacion.json")
TRACE_LOG = Path("traza_orquestacion.log")

PREGUNTA_DEMO = (
    "Investigá las reseñas de la Tarjeta Plus de BankLab, calculá su calificación "
    "promedio y el porcentaje de reseñas positivas, y calculá cuánto cuesta tenerla "
    "durante 3 años (cuota de emisión más comisión de mantenimiento mensual). "
    "Con eso, decime si la recomendarías."
)


def _linea_decision(d: dict[str, Any]) -> str:
    ev = d["evaluacion"]
    return (
        f"[Supervisor · decisión {d['paso']}] -> {d['siguiente']}\n"
        f"    razón: {d['razon']}\n"
        f"    rúbrica: fuentes={ev['datos_con_fuente']} | cálculos_por_analista="
        f"{ev['calculos_hechos_por_analista']} | responde_todo={ev['responde_toda_la_consulta']}"
        + (f"\n    instrucción: {d['instruccion']}" if d["instruccion"] else "")
    )


def _linea_aporte(c: dict[str, Any]) -> str:
    return (
        f"[{c['agente'].capitalize()}] herramientas: {', '.join(c['herramientas']) or 'ninguna'}\n"
        f"    resultado: {c['resultado']}"
    )


async def ejecutar(pregunta: str) -> dict[str, Any]:
    graph = build_graph(create_llm())
    lineas: list[str] = [f"Usuario: {pregunta}", ""]
    decisiones: list[dict[str, Any]] = []
    contribuciones: list[dict[str, Any]] = []
    respuesta, error = "", None

    try:
        async for update in graph.astream(
            estado_inicial(pregunta), {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
        ):
            for nodo, salida in update.items():
                for d in salida.get("decisiones", []):
                    decisiones.append(d)
                    lineas.append(_linea_decision(d))
                for c in salida.get("contribuciones", []):
                    contribuciones.append(c)
                    lineas.append(_linea_aporte(c))
                if nodo == "sintesis":
                    respuesta = str(salida["messages"][-1].content)
                    lineas.append(f"\n[Síntesis] Respuesta final:\n{respuesta}")
                if lineas:
                    print(lineas[-1] + "\n")
    except GraphRecursionError:
        error = f"Se alcanzó el recursion_limit ({RECURSION_LIMIT}) sin llegar a la síntesis."
        print(error)

    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "flujo": [d["siguiente"] for d in decisiones],
        "decisiones": decisiones,
        "contribuciones": contribuciones,
        "respuesta_final": respuesta,
        "error": error,
        "_log": "\n\n".join(lineas) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente.")
    parser.add_argument("--pregunta", default=PREGUNTA_DEMO)
    parser.add_argument("--graph", action="store_true", help="Imprime el diagrama Mermaid.")
    args = parser.parse_args()

    load_dotenv()
    try:
        if args.graph:
            print(build_graph(create_llm()).get_graph().draw_mermaid())
            return
        traza = asyncio.run(ejecutar(args.pregunta))
    except Exception as error:  # clave faltante, red, etc.: mensaje corto
        raise SystemExit(f"Error: {type(error).__name__}: {error}")

    log = traza.pop("_log")
    TRACE_JSON.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    TRACE_LOG.write_text(log, encoding="utf-8")
    print(f"Flujo de delegación: {' -> '.join(['supervisor'] + traza['flujo'])}")
    print(f"Traza guardada en {TRACE_JSON} y {TRACE_LOG}")


if __name__ == "__main__":
    main()
