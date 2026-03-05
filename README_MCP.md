# Integracion MCP + MLflow en Pipeline MLOps con Airflow

## Motivacion y Caso de Negocio

En un entorno de produccion, los pipelines de ML se ejecutan de forma periodica (diaria, semanal).
Cada ejecucion genera nuevos resultados: metricas de rendimiento, hiperparametros seleccionados,
estadisticas del dataset. Normalmente, un data scientist tiene que revisar manualmente estos
resultados -- abrir la base de datos, comparar metricas, escribir un resumen.

Esta integracion automatiza ese proceso: un asistente de IA (Claude) recibe acceso estructurado
a la infraestructura MLOps a traves de MCP y genera un reporte de analisis automaticamente
cada vez que se ejecuta el pipeline de entrenamiento.

### Valor que aporta

- **Eficiencia operativa**: Nadie tiene que revisar manualmente cada ejecucion del pipeline
- **Reportes accesibles**: Un reporte en lenguaje natural que cualquier stakeholder puede entender,
  sin necesidad de conocimientos tecnicos
- **Reportes estandarizados**: Misma estructura, misma profundidad, en cada ejecucion
- **Deteccion de problemas**: El sistema puede identificar caidas en accuracy, selecciones
  inusuales de hiperparametros, o cambios en la distribucion de datos
- **Trazabilidad completa**: MLflow registra cada experimento con parametros, metricas,
  artefactos y versiones del modelo

---

## Arquitectura

```
                    Airflow (DAG: 05_mcp_model_analysis)
                                    |
                                    v
                    +-------------------------------+
                    |        MCP Client             |
                    |   (dags/utils/mcp_client.py)  |
                    |                               |
                    |  1. Inicia servidor MCP        |
                    |  2. Descubre herramientas      |
                    |  3. Claude usa herramientas    |
                    |  4. Genera reporte             |
                    +-------------------------------+
                           |  stdio (stdin/stdout)
                           v
                    +-------------------------------+
                    |        MCP Server             |
                    |  (dags/utils/mcp_server.py)   |
                    |                               |
                    |  PostgreSQL Tools:            |
                    |  - get_experiment_results     |
                    |  - get_dataset_statistics     |
                    |  - get_best_model_info        |
                    |                               |
                    |  MLflow Tools:                |
                    |  - get_mlflow_experiments     |
                    |  - get_mlflow_model_versions  |
                    |  - get_mlflow_run_comparison  |
                    +-------------------------------+
                        |                   |
                        v                   v
               +----------------+  +------------------+
               |   PostgreSQL   |  |     MLflow       |
               |   Tables:      |  |  - Experiments   |
               |  - experiments |  |  - Runs          |
               |  - batch_data  |  |  - Model Registry|
               +----------------+  +------------------+
```

### Flujo de datos

1. El DAG de Airflow ejecuta la tarea `generate_analysis`
2. La tarea inicia el **cliente MCP**, que a su vez lanza el **servidor MCP** como subproceso
3. El cliente descubre las herramientas disponibles via `list_tools()` (6 herramientas)
4. En modo real: Claude decide que herramientas llamar y en que orden
5. En modo mock: el cliente llama todas las herramientas secuencialmente
6. Las herramientas consultan PostgreSQL y MLflow
7. Los resultados se ensamblan en un reporte markdown
8. El reporte se guarda en `data/reports/` y se muestra en los logs de Airflow

---

## Por que MCP (Model Context Protocol)

MCP es un protocolo abierto creado por Anthropic que estandariza como los LLMs
interactuan con herramientas y fuentes de datos externas.

### Ventajas frente a llamadas directas

| Aspecto | Sin MCP | Con MCP |
|---|---|---|
| Agregar nueva fuente de datos | Reescribir el DAG | Agregar un `@mcp.tool()` al servidor |
| Reutilizacion | Codigo acoplado al DAG | El servidor se puede usar con Claude Desktop, otros clientes |
| Descubrimiento | Herramientas hardcodeadas | El LLM descubre herramientas dinamicamente |
| Estandarizacion | Cada proyecto implementa su propio esquema | Protocolo estandar con ecosistema creciente |

### Transporte stdio

Se utiliza transporte **stdio** (stdin/stdout) porque el servidor y el cliente corren
en el mismo contenedor Docker. No se necesita configuracion de red ni puertos adicionales.
El servidor se inicia como subproceso del cliente y se cierra al terminar la tarea.

---

## Integracion con MLflow

### Que aporta MLflow al proyecto

El pipeline ML original solo guardaba resultados en una tabla plana de PostgreSQL.
Con MLflow, cada ejecucion ahora registra:

- **Parametros**: cv_folds, logreg_maxiter, max_pca_components, best_logreg_c, best_pca_components
- **Metricas**: test_set_accuracy, best_cv_score
- **Artefactos**: confusion matrix (CSV), classification report (TXT)
- **Modelo versionado**: registrado en el Model Registry con version automatica

### Model Registry

El Model Registry permite gestionar el ciclo de vida del modelo:
- Cada entrenamiento crea una nueva version
- Las versiones se pueden promover a Staging o Production
- Se mantiene un historial completo de todas las versiones

### Herramientas MCP para MLflow

| Herramienta | Descripcion |
|---|---|
| `get_mlflow_experiments` | Lista runs con parametros y metricas |
| `get_mlflow_model_versions` | Consulta el Model Registry (versiones, stage, estado) |
| `get_mlflow_run_comparison` | Compara los mejores runs por accuracy |

---

## Componentes

| Archivo | Descripcion |
|---|---|
| `dags/mcp_model_analysis.py` | DAG de Airflow (05_mcp_model_analysis) con dos tareas |
| `dags/utils/mcp_server.py` | Servidor MCP con 6 herramientas (3 PostgreSQL + 3 MLflow) |
| `dags/utils/mcp_client.py` | Cliente MCP que conecta al servidor y usa Claude (o mock) |
| `dags/utils/mlflow_config.py` | Configuracion centralizada de MLflow y base de datos |
| `dags/utils/simulate_experiments.py` | Script de simulacion con datos incrementales |
| `.env.example` | Plantilla de variables de entorno |
| `README_MCP.md` | Este documento |

### Archivos modificados

| Archivo | Cambio |
|---|---|
| `dags/utils/experiment.py` | Logging a MLflow (parametros, metricas, artefactos, model registry) |
| `dags/sql/create_experiments.sql` | Agregada columna `mlflow_run_id` |
| `docker-compose.yaml` | Variables de entorno centralizadas (`DATABASE_URL`, `MLFLOW_TRACKING_URI`) |
| `.gitignore` | Agregado `.env` para proteger credenciales |

---

## Seguridad y Gestion de Credenciales

### Principio: las credenciales nunca se commitean

Las credenciales (API keys, connection strings) se gestionan a traves de variables de entorno:

1. **`.env.example`** -- plantilla que se commitea como referencia, con valores de ejemplo
2. **`.env`** -- archivo real con credenciales, incluido en `.gitignore`
3. **`docker-compose.yaml`** -- referencia variables de entorno con fallback a valores por defecto

### En produccion se usaria

- **HashiCorp Vault** o **AWS Secrets Manager** -- las credenciales se piden en tiempo de ejecucion
- **Kubernetes Secrets** -- se inyectan como variables de entorno desde el cluster
- **Airflow Connections** -- para credenciales de bases de datos dentro del ecosistema Airflow
- **Rotacion automatica** -- las credenciales se cambian periodicamente sin intervencion humana

---

## Como Ejecutar

### Prerequisitos

1. Docker y Docker Compose instalados
2. El pipeline ML (`04_ml_pipeline`) debe haber corrido al menos una vez
   para que haya datos en las tablas `experiments` y `batch_data`

### Pasos

```bash
# 1. Copiar plantilla de variables de entorno
cp .env.example .env
# Editar .env con valores reales (API key si se desea modo real)

# 2. Reiniciar contenedores (instala nuevos paquetes pip)
docker compose down
docker compose up -d

# 3. Acceder a Airflow UI
#    URL: http://localhost:8080
#    Usuario: airflow / Password: airflow

# 4. Ejecutar primero el pipeline ML (si no se ha ejecutado antes)
#    Activar y triggear el DAG: 04_ml_pipeline

# 5. (Opcional) Ejecutar simulacion para generar historial de experimentos
#    docker exec -it <airflow-scheduler> python /opt/airflow/dags/utils/simulate_experiments.py

# 6. Ejecutar el DAG de analisis MCP
#    Activar y triggear el DAG: 05_mcp_model_analysis

# 7. Ver resultados
#    - En Airflow UI: click en la tarea generate_analysis -> Logs
#    - En MLflow UI: http://localhost:5000
#    - En el host: ls data/reports/
```

---

## Modo Mock vs Modo Real

### Modo Mock (por defecto)

Cuando no hay `ANTHROPIC_API_KEY` configurada, el sistema funciona en modo mock:

- El servidor MCP se inicia y las herramientas se ejecutan **realmente**
  (consultas reales a PostgreSQL y MLflow)
- El protocolo MCP se ejerce completo (stdio transport, list_tools, call_tool)
- El reporte se ensambla con una plantilla que incluye los datos reales

Este modo demuestra toda la arquitectura MCP sin costo de API.

### Modo Real (con API key)

```bash
# Editar .env y agregar la API key
ANTHROPIC_API_KEY=sk-ant-api03-tu-key-aqui

# Reiniciar contenedores
docker compose down && docker compose up -d
```

En modo real:
- Claude recibe las 6 herramientas disponibles y decide cuales usar
- El LLM genera un analisis inteligente con insights y recomendaciones
- El flujo MCP es identico, pero con razonamiento de IA real
- Claude puede cruzar datos de PostgreSQL y MLflow para un analisis mas completo
