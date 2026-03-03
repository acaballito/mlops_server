"""
Servidor MCP para infraestructura MLOps.
Expone herramientas para consultar resultados de experimentos,
estadisticas del dataset y rendimiento del modelo desde PostgreSQL.

Ejecutar standalone: python mcp_server.py
O conectar via cliente MCP (stdio transport).
"""
from mcp.server.fastmcp import FastMCP
import pandas as pd
from sqlalchemy import create_engine, text

mcp = FastMCP("MLOps Analysis Server")

DB_ENGINE = "postgresql+psycopg2://airflow:airflow@postgres/airflow"


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


if __name__ == "__main__":
    mcp.run()
