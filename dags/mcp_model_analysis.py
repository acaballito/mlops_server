"""
DAG: 05_mcp_model_analysis
Conecta al servidor MCP de MLOps para producir un reporte
automatizado de analisis de resultados de experimentos ML.
"""
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os


default_args = {
    'owner': 'Data Scientist',
    'email_on_failure': False,
    'email': ['ds@mymail.com'],
    'start_date': datetime(2025, 10, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


def generate_analysis_report(**kwargs):
    """Ejecuta el cliente MCP para generar el reporte de analisis."""
    from utils.mcp_client import run_mcp_analysis

    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    server_path = "/opt/airflow/dags/utils/mcp_server.py"

    mode = "REAL (Claude)" if api_key else "MOCK (sin LLM)"
    print(f"[DAG] Starting MCP analysis in {mode} mode...")

    report = run_mcp_analysis(server_path, api_key)

    # Guardar reporte en el volumen compartido de datos
    report_dir = "/opt/airflow/data/reports"
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{report_dir}/analysis_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(report)

    kwargs["ti"].xcom_push(key="report_path", value=report_path)
    print(f"[DAG] Report saved to: {report_path}")


def print_report_summary(**kwargs):
    """Lee e imprime un resumen del reporte en los logs de Airflow."""
    report_path = kwargs["ti"].xcom_pull(
        task_ids="generate_analysis", key="report_path"
    )
    with open(report_path, "r") as f:
        content = f.read()
    print("=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)
    print(content[:3000])
    if len(content) > 3000:
        print(f"\n... [truncated, full report at {report_path}]")
    print("=" * 60)


with DAG(
    "05_mcp_model_analysis",
    description='AI-powered analysis of ML experiment results via MCP',
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=["mcp", "analysis"],
) as dag:

    # Tarea 1: Generar reporte de analisis usando MCP + Claude/Mock
    generate_analysis = PythonOperator(
        task_id='generate_analysis',
        python_callable=generate_analysis_report,
        execution_timeout=timedelta(minutes=5),
    )

    # Tarea 2: Imprimir resumen del reporte en los logs de Airflow
    print_summary = PythonOperator(
        task_id='print_report_summary',
        python_callable=print_report_summary,
    )

    generate_analysis >> print_summary
