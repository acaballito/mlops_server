"""
Configuracion centralizada para MLflow y conexion a base de datos.
Elimina valores hardcodeados del resto del codigo -- las URIs se leen
de variables de entorno con fallback a los valores por defecto de Docker.
"""
import os
import mlflow

# Conexion a base de datos (usada por mcp_server, simulate_experiments, etc.)
DB_ENGINE = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://airflow:airflow@postgres/airflow",
)

# MLflow
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://mlserver-mlflow:5000",
)
MLFLOW_EXPERIMENT_NAME = "breast_cancer_classification"
MLFLOW_MODEL_NAME = "breast_cancer_classifier"


def setup_mlflow():
    """Configura el tracking URI y crea/obtiene el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
