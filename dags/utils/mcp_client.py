"""
Cliente MCP que conecta al servidor MLOps y utiliza Claude (o modo mock)
para analizar resultados de experimentos via tool use.
"""
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ANALYSIS_PROMPT = """Eres un ingeniero de ML analizando resultados de un pipeline de clasificacion
de cancer de mama. El pipeline usa el dataset breast cancer de sklearn con
Pipeline(StandardScaler -> PCA -> LogisticRegression) entrenado via GridSearchCV.
Los experimentos se registran en PostgreSQL y MLflow.

Realiza un analisis completo:
1. Obtener los resultados de experimentos de PostgreSQL
2. Obtener las estadisticas del dataset
3. Obtener la informacion del mejor modelo de PostgreSQL
4. Consultar MLflow para el historial completo de experimentos con parametros y metricas
5. Revisar el MLflow Model Registry para las versiones registradas del modelo
6. Comparar los mejores runs en MLflow
7. Leer los artefactos (confusion matrix y classification report) de los runs con
   mejor y peor rendimiento para entender los patrones de error
8. Ejecutar un analisis de drift para verificar si el rendimiento es estable

Genera el reporte EN ESPANOL con DOS secciones claramente diferenciadas:

---

## PARTE 1: RESUMEN PARA DIRECCION

Escrito para directivos y stakeholders no tecnicos. Maximo 1 pagina.
- Situacion actual del modelo en una frase
- Nivel de fiabilidad: que tan confiable es el modelo para tomar decisiones clinicas
- Riesgos identificados: en lenguaje de negocio, sin tecnicismos
  (ej: "el modelo falla en detectar 1 de cada 100 tumores malignos")
- Acciones recomendadas: que debe aprobar o decidir la direccion
- Impacto si no se actua: que pasa si no se toman las acciones recomendadas

NO usar terminos como "PCA", "regularizacion", "hiperparametros", "GridSearchCV",
"confusion matrix", "F1-score", "cross-validation" ni ningun otro tecnicismo.
Usar lenguaje claro y directo que cualquier persona pueda entender.

---

## PARTE 2: ANALISIS TECNICO PARA EQUIPO DE MLOPS Y DATA SCIENCE

Escrito para el equipo tecnico. Detallado y especifico con numeros reales.
- Historial de experimentos: tendencias, evolucion de hiperparametros
- Analisis de MLflow: insights de metricas y parametros
- Estado del Model Registry: versiones, stages, que promover
- Analisis de errores: falsos positivos vs falsos negativos, contexto medico
  (un falso negativo significa no detectar un tumor maligno)
- Analisis de drift: estabilidad del rendimiento, cambios en hiperparametros
- Analisis del mejor modelo: por que funciona la mejor configuracion
- Caracteristicas del dataset: observaciones sobre distribuciones
- Recomendaciones tecnicas: pasos concretos para mejorar el modelo

Formato markdown limpio. Usa numeros reales de los datos."""


async def run_analysis(server_script_path: str, anthropic_api_key: str = None) -> str:
    """
    Conecta al servidor MCP, usa Claude (o mock) para analizar datos MLOps,
    y retorna el reporte de analisis como string.
    """
    exit_stack = AsyncExitStack()

    try:
        # 1. Conectar al servidor MCP via stdio
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env=None,  # heredar entorno del proceso padre
        )
        # Abrir un archivo real para stderr del subproceso MCP.
        # Necesario porque Airflow LocalExecutor usa fork() y
        # sys.stderr pierde su file descriptor real.
        errlog = open(os.path.join("/tmp", "mcp_server_stderr.log"), "w")
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params, errlog=errlog)
        )
        read_stream, write_stream = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        # 2. Descubrir herramientas disponibles
        tools_response = await session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools_response.tools
        ]

        print(f"[MCP] Connected to server. Available tools: "
              f"{[t['name'] for t in available_tools]}")

        # 3. Elegir modo: real (con API key) o mock (sin API key)
        if anthropic_api_key:
            report = await _run_with_claude(
                session, available_tools, anthropic_api_key
            )
        else:
            report = await _run_mock_analysis(session, available_tools)

        return report

    finally:
        await exit_stack.aclose()


async def _run_with_claude(
    session: ClientSession, available_tools: list, api_key: str
) -> str:
    """Modo real: Claude decide que herramientas llamar via tool use."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": ANALYSIS_PROMPT}]

    max_iterations = 10
    iteration = 0
    final_text = []

    while iteration < max_iterations:
        iteration += 1
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages,
            tools=available_tools,
        )

        assistant_content = []
        tool_calls_made = False

        for block in response.content:
            assistant_content.append(block)
            if block.type == "text":
                final_text.append(block.text)
            elif block.type == "tool_use":
                tool_calls_made = True
                print(f"[MCP] Claude calling tool: {block.name}({block.input})")

                result = await session.call_tool(block.name, block.input)
                result_text = ""
                for content_block in result.content:
                    if hasattr(content_block, "text"):
                        result_text += content_block.text

                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    ],
                })
                assistant_content = []

        if not tool_calls_made or response.stop_reason == "end_turn":
            break

    return "\n".join(final_text)


async def _run_mock_analysis(
    session: ClientSession, available_tools: list
) -> str:
    """
    Modo mock: llama todas las herramientas MCP reales y ensambla
    un reporte estructurado sin necesidad de un LLM.
    Demuestra el flujo MCP completo (server + client + tool calls).
    """
    print("[MCP] Running in MOCK mode (no API key). "
          "Calling all tools via MCP protocol...")

    # --- Herramientas PostgreSQL ---
    print("[MCP] Calling tool: get_experiment_results")
    experiments_result = await session.call_tool(
        "get_experiment_results", {"limit": 10}
    )
    experiments_data = _extract_text(experiments_result)

    print("[MCP] Calling tool: get_dataset_statistics")
    stats_result = await session.call_tool("get_dataset_statistics", {})
    stats_data = _extract_text(stats_result)

    print("[MCP] Calling tool: get_best_model_info")
    model_result = await session.call_tool("get_best_model_info", {})
    model_data = _extract_text(model_result)

    # --- Herramientas MLflow ---
    print("[MCP] Calling tool: get_mlflow_experiments")
    mlflow_exp_result = await session.call_tool("get_mlflow_experiments", {})
    mlflow_exp_data = _extract_text(mlflow_exp_result)

    print("[MCP] Calling tool: get_mlflow_model_versions")
    mlflow_versions_result = await session.call_tool(
        "get_mlflow_model_versions", {}
    )
    mlflow_versions_data = _extract_text(mlflow_versions_result)

    print("[MCP] Calling tool: get_mlflow_run_comparison")
    mlflow_comparison_result = await session.call_tool(
        "get_mlflow_run_comparison", {}
    )
    mlflow_comparison_data = _extract_text(mlflow_comparison_result)

    print("[MCP] Calling tool: get_mlflow_run_artifacts (best run)")
    artifacts_result = await session.call_tool(
        "get_mlflow_run_artifacts", {"run_id": "26844eb5"}
    )
    artifacts_data = _extract_text(artifacts_result)

    print("[MCP] Calling tool: get_experiment_drift_analysis")
    drift_result = await session.call_tool(
        "get_experiment_drift_analysis", {}
    )
    drift_data = _extract_text(drift_result)

    # Ensamblar reporte con los datos reales obtenidos via MCP
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Reporte de Analisis ML - Generado via MCP
**Fecha**: {timestamp}
**Modo**: Mock (sin LLM) - Datos obtenidos via protocolo MCP

---

## Resumen Ejecutivo

Este reporte fue generado automaticamente utilizando el protocolo MCP
(Model Context Protocol). El cliente MCP conecto al servidor MLOps,
descubrio las herramientas disponibles ({len(available_tools)} tools),
y las ejecuto para recopilar datos de PostgreSQL y MLflow.

En un escenario con API key configurada, Claude analizaria estos datos
y generaria insights automaticos sobre el rendimiento del modelo.

---

## Resultados de Experimentos (PostgreSQL)

Datos obtenidos via `get_experiment_results`:

{experiments_data}

---

## Informacion del Mejor Modelo

Datos obtenidos via `get_best_model_info`:

{model_data}

---

## Estadisticas del Dataset

Datos obtenidos via `get_dataset_statistics`:

{stats_data}

---

## MLflow - Historial de Experimentos

Datos obtenidos via `get_mlflow_experiments`:

{mlflow_exp_data}

---

## MLflow - Registro de Modelos

Datos obtenidos via `get_mlflow_model_versions`:

{mlflow_versions_data}

---

## MLflow - Comparacion de Runs

Datos obtenidos via `get_mlflow_run_comparison`:

{mlflow_comparison_data}

---

## Analisis de Errores del Modelo

Datos obtenidos via `get_mlflow_run_artifacts`:

{artifacts_data}

---

## Analisis de Drift

Datos obtenidos via `get_experiment_drift_analysis`:

{drift_data}

---

## Herramientas MCP Utilizadas

| Herramienta | Descripcion |
|---|---|
"""
    for tool in available_tools:
        report += f"| `{tool['name']}` | {tool['description'][:80]}... |\n"

    report += """
---

## Nota

Este reporte demuestra el flujo completo del protocolo MCP:
1. El cliente MCP inicio el servidor como subproceso (stdio transport)
2. Descubrio las herramientas disponibles via `list_tools()`
3. Ejecuto cada herramienta via `call_tool()` (consultas reales a PostgreSQL y MLflow)
4. Ensamblo los resultados en este reporte

Con una API key de Anthropic configurada (`ANTHROPIC_API_KEY`), Claude
utilizaria estas mismas herramientas de forma autonoma para generar
un analisis inteligente con insights y recomendaciones.
"""
    return report


def _extract_text(result) -> str:
    """Extrae texto de la respuesta de una herramienta MCP."""
    text_parts = []
    for content_block in result.content:
        if hasattr(content_block, "text"):
            text_parts.append(content_block.text)
    return "\n".join(text_parts)


def run_mcp_analysis(server_script_path: str, anthropic_api_key: str = None) -> str:
    """Wrapper sincrono para la funcion de analisis asincrona."""
    return asyncio.run(run_analysis(server_script_path, anthropic_api_key))
