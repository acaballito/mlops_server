"""
Servidor MCP para infraestructura MLOps.
Expone herramientas para consultar resultados de experimentos,
estadisticas del dataset y rendimiento del modelo desde PostgreSQL y MLflow.

Ejecutar standalone: python mcp_server.py
O conectar via cliente MCP (stdio transport).
"""
import os
import sys

from mcp.server.fastmcp import FastMCP
import pandas as pd
from sqlalchemy import create_engine, text
import mlflow
from mlflow.tracking import MlflowClient

# Asegurar que el directorio padre (dags/) esta en el path
# para que los imports funcionen tanto desde Airflow como standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mlflow_config import (
    DB_ENGINE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME,
)

mcp = FastMCP("MLOps Analysis Server")


# --- Herramientas PostgreSQL (existentes) ---

@mcp.tool()
def get_experiment_results(limit: int = 10) -> str:
    """Query the latest experiment results from the experiments table.
    Returns experiment_id, datetime, hyperparameters (cv_folds, logreg_maxiter,
    max_pca_components, best_logreg_c, best_pca_components), and test_set_accuracy.
    """
    engine = create_engine(DB_ENGINE)
    query = text(
        "SELECT * FROM public.experiments ORDER BY experiment_id DESC LIMIT :lim"
    )
    df = pd.read_sql(query, engine, params={"lim": limit})
    engine.dispose()
    if df.empty:
        return "No experiments found in the database. The ML pipeline may not have run yet."
    return df.to_markdown(index=False)


@mcp.tool()
def get_dataset_statistics() -> str:
    """Get descriptive statistics of the breast cancer dataset stored in batch_data.
    Returns count, mean, std, min, max for all features plus the label column.
    """
    engine = create_engine(DB_ENGINE)
    try:
        df = pd.read_sql("SELECT * FROM public.batch_data", engine)
    except Exception as e:
        return f"Could not read batch_data table: {e}"
    finally:
        engine.dispose()
    if df.empty:
        return "The batch_data table is empty. The ML pipeline may not have run yet."
    stats = df.describe().round(3)
    return stats.to_markdown()


@mcp.tool()
def get_best_model_info() -> str:
    """Get information about the best model from the most recent experiment.
    Compares the latest experiment with the historical best.
    """
    engine = create_engine(DB_ENGINE)
    query = text("SELECT * FROM public.experiments ORDER BY experiment_id DESC")
    df = pd.read_sql(query, engine)
    engine.dispose()
    if df.empty:
        return "No experiments found."
    latest = df.iloc[0]
    best_ever = df.loc[df["test_set_accuracy"].idxmax()]
    result_lines = [
        "## Latest Experiment",
        f"- Experiment ID: {latest['experiment_id']}",
        f"- Date: {latest['experiment_datetime']}",
        f"- Best LogReg C: {latest['best_logreg_c']}",
        f"- Best PCA Components: {int(latest['best_pca_components'])}",
        f"- Test Accuracy: {latest['test_set_accuracy']}",
        f"- CV Folds: {int(latest['cv_folds'])}",
        "",
        "## Historical Best",
        f"- Experiment ID: {best_ever['experiment_id']}",
        f"- Date: {best_ever['experiment_datetime']}",
        f"- Test Accuracy: {best_ever['test_set_accuracy']}",
        f"- PCA Components: {int(best_ever['best_pca_components'])}",
        f"- LogReg C: {best_ever['best_logreg_c']}",
        "",
        f"## Total Experiments Run: {len(df)}",
    ]
    return "\n".join(result_lines)


# --- Herramientas MLflow (nuevas) ---

@mcp.tool()
def get_mlflow_experiments(
    experiment_name: str = "breast_cancer_classification", max_runs: int = 10
) -> str:
    """List MLflow experiments and their recent runs.
    Returns experiment metadata and run details including parameters, metrics, and status.
    Use this to see the full history of model training tracked in MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return f"No MLflow experiment found with name '{experiment_name}'."

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_runs,
        )

        if not runs:
            return f"Experiment '{experiment_name}' exists but has no runs yet."

        rows = []
        for run in runs:
            rows.append({
                "run_id": run.info.run_id[:8],
                "status": run.info.status,
                "start_time": pd.Timestamp(
                    run.info.start_time, unit="ms"
                ).strftime("%Y-%m-%d %H:%M"),
                **{f"param_{k}": v for k, v in run.data.params.items()},
                **{
                    f"metric_{k}": round(float(v), 4)
                    for k, v in run.data.metrics.items()
                },
            })

        df = pd.DataFrame(rows)
        header = (
            f"## MLflow Experiment: {experiment_name}\n"
            f"- Experiment ID: {experiment.experiment_id}\n"
            f"- Total runs shown: {len(runs)}\n\n"
        )
        return header + df.to_markdown(index=False)
    except Exception as e:
        return f"Error querying MLflow: {e}"


@mcp.tool()
def get_mlflow_model_versions(
    model_name: str = "breast_cancer_classifier",
) -> str:
    """Query the MLflow Model Registry for registered versions of a model.
    Returns version number, stage (None/Staging/Production/Archived),
    creation timestamp, run_id, and current status.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return f"No registered model found with name '{model_name}'."

        rows = []
        for v in versions:
            rows.append({
                "version": v.version,
                "stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id[:8] if v.run_id else "N/A",
                "created": pd.Timestamp(
                    v.creation_timestamp, unit="ms"
                ).strftime("%Y-%m-%d %H:%M"),
                "description": v.description or "",
            })

        df = pd.DataFrame(rows)
        header = (
            f"## MLflow Model Registry: {model_name}\n"
            f"- Total versions: {len(versions)}\n\n"
        )
        return header + df.to_markdown(index=False)
    except Exception as e:
        return f"Error querying MLflow Model Registry: {e}"


@mcp.tool()
def get_mlflow_run_comparison(
    experiment_name: str = "breast_cancer_classification", top_n: int = 5
) -> str:
    """Compare metrics across the top N best MLflow runs by test_set_accuracy.
    Returns a side-by-side comparison of parameters and metrics for the best runs,
    highlighting which configuration achieved the highest performance.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return f"No MLflow experiment found with name '{experiment_name}'."

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_set_accuracy DESC"],
            max_results=top_n,
        )

        if not runs:
            return "No runs found to compare."

        rows = []
        for rank, run in enumerate(runs, 1):
            rows.append({
                "rank": rank,
                "run_id": run.info.run_id[:8],
                "test_accuracy": run.data.metrics.get(
                    "test_set_accuracy", "N/A"
                ),
                "best_cv_score": run.data.metrics.get("best_cv_score", "N/A"),
                "pca_components": run.data.params.get(
                    "best_pca_components", "N/A"
                ),
                "logreg_c": run.data.params.get("best_logreg_c", "N/A"),
                "cv_folds": run.data.params.get("cv_folds", "N/A"),
            })

        df = pd.DataFrame(rows)

        accuracies = [
            r.data.metrics.get("test_set_accuracy", 0)
            for r in runs
            if "test_set_accuracy" in r.data.metrics
        ]
        summary = ""
        if accuracies:
            summary = (
                f"\n\n### Summary Statistics\n"
                f"- Best accuracy: {max(accuracies):.4f}\n"
                f"- Worst accuracy (in top {top_n}): {min(accuracies):.4f}\n"
                f"- Spread: {max(accuracies) - min(accuracies):.4f}\n"
            )

        header = f"## Top {top_n} Runs by Test Accuracy\n\n"
        return header + df.to_markdown(index=False) + summary
    except Exception as e:
        return f"Error comparing MLflow runs: {e}"


@mcp.tool()
def get_mlflow_run_artifacts(run_id: str) -> str:
    """Read the confusion matrix and classification report artifacts from a specific MLflow run.
    Provide the full or partial run_id. Returns the confusion matrix as a table and
    the full classification report with precision, recall, and f1-score per class.
    Use this to understand model errors: false positives vs false negatives.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Buscar el run completo si se paso un ID parcial
        if len(run_id) < 32:
            experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                return "No experiment found."
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=100,
            )
            matching = [r for r in runs if r.info.run_id.startswith(run_id)]
            if not matching:
                return f"No run found matching '{run_id}'."
            full_run_id = matching[0].info.run_id
        else:
            full_run_id = run_id

        # Descargar artefactos a directorio temporal
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_path = client.download_artifacts(
                full_run_id, "", tmpdir
            )

            result_parts = [f"## Artifacts for run {full_run_id[:8]}\n"]

            # Leer confusion matrix
            cm_path = os.path.join(artifacts_path, "confusion_matrix.csv")
            if os.path.exists(cm_path):
                cm_df = pd.read_csv(cm_path)
                cm_df.index = ["Actual Benign", "Actual Malignant"]
                cm_df.columns = ["Predicted Benign", "Predicted Malignant"]
                result_parts.append("### Confusion Matrix\n")
                result_parts.append(cm_df.to_markdown())

                # Calcular falsos positivos y falsos negativos
                tn, fp = cm_df.iloc[0]
                fn, tp = cm_df.iloc[1]
                result_parts.append(f"\n- True Positives (correctly detected malignant): {int(tp)}")
                result_parts.append(f"- True Negatives (correctly detected benign): {int(tn)}")
                result_parts.append(f"- False Positives (benign classified as malignant): {int(fp)}")
                result_parts.append(f"- False Negatives (malignant missed as benign): {int(fn)}")
                if fn > 0:
                    result_parts.append(
                        f"\n**WARNING**: {int(fn)} malignant case(s) missed. "
                        f"In medical contexts, false negatives are critical."
                    )
            else:
                result_parts.append("Confusion matrix artifact not found.")

            # Leer classification report
            cr_path = os.path.join(artifacts_path, "classification_report.txt")
            if os.path.exists(cr_path):
                with open(cr_path, "r") as f:
                    cr_text = f.read()
                result_parts.append("\n### Classification Report\n")
                result_parts.append(f"```\n{cr_text}\n```")
            else:
                result_parts.append("Classification report artifact not found.")

        return "\n".join(result_parts)
    except Exception as e:
        return f"Error reading artifacts: {e}"


@mcp.tool()
def get_experiment_drift_analysis(
    experiment_name: str = "breast_cancer_classification",
) -> str:
    """Analyze data drift by comparing metrics across experiments ordered by time.
    Detects if model performance is degrading or if hyperparameter selections
    are shifting significantly, which may indicate changes in the underlying data.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return f"No MLflow experiment found with name '{experiment_name}'."

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time ASC"],
            max_results=50,
        )

        if len(runs) < 2:
            return "Need at least 2 runs to analyze drift."

        # Construir serie temporal de metricas
        rows = []
        for run in runs:
            accuracy = run.data.metrics.get("test_set_accuracy")
            cv_score = run.data.metrics.get("best_cv_score")
            n_samples = run.data.params.get("n_samples", "N/A")
            pca = run.data.params.get("best_pca_components", "N/A")
            c_val = run.data.params.get("best_logreg_c", "N/A")

            if accuracy is not None:
                rows.append({
                    "run_id": run.info.run_id[:8],
                    "date": pd.Timestamp(
                        run.info.start_time, unit="ms"
                    ).strftime("%Y-%m-%d"),
                    "n_samples": n_samples,
                    "accuracy": round(accuracy, 4),
                    "cv_score": round(cv_score, 4) if cv_score else "N/A",
                    "pca_components": pca,
                    "logreg_c": c_val,
                })

        if not rows:
            return "No runs with metrics found."

        df = pd.DataFrame(rows)

        # Analisis de tendencia
        accuracies = [r["accuracy"] for r in rows]
        first_half = accuracies[:len(accuracies)//2]
        second_half = accuracies[len(accuracies)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        # Detectar cambios en hiperparametros
        pca_values = [r["pca_components"] for r in rows if r["pca_components"] != "N/A"]
        c_values = [r["logreg_c"] for r in rows if r["logreg_c"] != "N/A"]

        result_parts = ["## Drift Analysis\n"]
        result_parts.append("### Performance Over Time\n")
        result_parts.append(df.to_markdown(index=False))

        result_parts.append("\n\n### Trend Summary\n")
        result_parts.append(f"- First half avg accuracy: {avg_first:.4f}")
        result_parts.append(f"- Second half avg accuracy: {avg_second:.4f}")

        delta = avg_second - avg_first
        if delta > 0.02:
            result_parts.append(
                f"- **Trend: IMPROVING** (+{delta:.4f}). "
                f"Performance is increasing over time."
            )
        elif delta < -0.02:
            result_parts.append(
                f"- **Trend: DEGRADING** ({delta:.4f}). "
                f"Performance is declining -- investigate potential data drift."
            )
        else:
            result_parts.append(
                f"- **Trend: STABLE** ({delta:+.4f}). "
                f"Performance is consistent across experiments."
            )

        # Analisis de variabilidad de hiperparametros
        if len(set(pca_values)) > 1:
            result_parts.append(
                f"\n### Hyperparameter Stability\n"
                f"- PCA components range: {min(pca_values)} to {max(pca_values)} "
                f"({len(set(pca_values))} distinct values)"
            )
        if len(set(c_values)) > 1:
            result_parts.append(
                f"- LogReg C range: {min(c_values)} to {max(c_values)} "
                f"({len(set(c_values))} distinct values)"
            )

        # Gap CV vs test
        cv_scores = [r["cv_score"] for r in rows if r["cv_score"] != "N/A"]
        if cv_scores:
            gaps = [
                abs(r["accuracy"] - r["cv_score"])
                for r in rows
                if r["cv_score"] != "N/A"
            ]
            avg_gap = sum(gaps) / len(gaps)
            result_parts.append(
                f"\n### Generalization Gap\n"
                f"- Average |test - cv| gap: {avg_gap:.4f}"
            )
            if avg_gap > 0.03:
                result_parts.append(
                    "- **WARNING**: Large gap between CV and test performance "
                    "suggests potential overfitting or data distribution issues."
                )
            else:
                result_parts.append(
                    "- Gap is within normal range -- model generalizes well."
                )

        return "\n".join(result_parts)
    except Exception as e:
        return f"Error analyzing drift: {e}"


if __name__ == "__main__":
    mcp.run()
