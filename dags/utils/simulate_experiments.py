"""
Simula un escenario real donde el dataset crece con el tiempo.
Ejecuta el pipeline ML con subconjuntos incrementales de datos
(150, 200, 250, 300, 350, 400, 450, 500, 569 muestras)
y guarda cada resultado en PostgreSQL y MLflow.
"""
import os
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sqlalchemy import create_engine

import mlflow
import mlflow.sklearn
from utils.mlflow_config import DB_ENGINE, setup_mlflow, MLFLOW_MODEL_NAME

SAMPLE_SIZES = [150, 200, 250, 300, 350, 400, 450, 500, 569]


def run_simulation():
    data = load_breast_cancer()
    full_df = pd.DataFrame(data.data, columns=data.feature_names)
    full_df["label"] = data.target

    engine = create_engine(DB_ENGINE)

    # Configurar MLflow
    setup_mlflow()

    # Simular que cada ejecucion ocurre con una semana de diferencia
    base_date = datetime(2026, 1, 6)

    for i, n_samples in enumerate(SAMPLE_SIZES):
        # Tomar subconjunto aleatorio del dataset
        subset = full_df.sample(n=n_samples, random_state=None)
        X = subset.iloc[:, :-1]
        y = subset["label"]

        # Split train/test
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3
        )

        # Pipeline con grid search mas amplio
        pipe = Pipeline([
            ("std_scaler", StandardScaler()),
            ("pca", PCA()),
            ("log_reg", LogisticRegression(max_iter=1000)),
        ])

        params = {
            "pca__n_components": list(range(1, min(30, n_samples))),
            "log_reg__C": np.logspace(-1, 1, 5),
        }

        grid_search = GridSearchCV(pipe, params, cv=3)
        grid_search.fit(x_train, y_train)

        best_c = round(grid_search.best_params_["log_reg__C"], 4)
        best_pca = grid_search.best_params_["pca__n_components"]

        y_pred = grid_search.best_estimator_.predict(x_test)
        accuracy = round(accuracy_score(y_test, y_pred), 3)

        # Simular fecha incrementada semanalmente
        exp_date = base_date + timedelta(weeks=i)
        exp_datetime = exp_date.strftime("%d-%m-%Y_%H:%M:%S")

        # --- Registrar en MLflow ---
        with mlflow.start_run() as run:
            mlflow.log_param("cv_folds", 3)
            mlflow.log_param("logreg_maxiter", 1000)
            mlflow.log_param("max_pca_components", 30)
            mlflow.log_param("test_split_ratio", 0.3)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("best_logreg_c", best_c)
            mlflow.log_param("best_pca_components", best_pca)

            mlflow.log_metric("test_set_accuracy", accuracy)
            mlflow.log_metric("best_cv_score", round(grid_search.best_score_, 3))

            # Artefactos: confusion matrix y classification report
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)

            with tempfile.TemporaryDirectory() as tmpdir:
                cm_path = os.path.join(tmpdir, "confusion_matrix.csv")
                pd.DataFrame(cm).to_csv(cm_path, index=False)
                mlflow.log_artifact(cm_path)

                cr_path = os.path.join(tmpdir, "classification_report.txt")
                with open(cr_path, "w") as f:
                    f.write(cr)
                mlflow.log_artifact(cr_path)

            # Registrar modelo en Model Registry
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME,
            )

            mlflow_run_id = run.info.run_id

        # --- Guardar en PostgreSQL ---
        exp_info = pd.DataFrame(
            [[exp_datetime, 3, 1000, 30, best_c, best_pca, accuracy,
              mlflow_run_id]],
            columns=[
                "experiment_datetime", "cv_folds", "logreg_maxiter",
                "max_pca_components", "best_logreg_c",
                "best_pca_components", "test_set_accuracy",
                "mlflow_run_id",
            ],
        )
        exp_info.to_sql(
            "experiments", engine, schema="public",
            if_exists="append", index=False,
        )
        print(
            f"[SIM] n={n_samples:>3} | PCA={best_pca:>2} | "
            f"C={best_c:.4f} | accuracy={accuracy:.3f} | "
            f"mlflow_run={mlflow_run_id[:8]} | date={exp_datetime}"
        )

    # Guardar el dataset completo en batch_data
    full_df.to_sql(
        "batch_data", engine, schema="public",
        if_exists="replace", index=False,
    )
    print(f"\n[SIM] Simulation complete: {len(SAMPLE_SIZES)} experiments saved "
          f"to PostgreSQL and MLflow.")
    engine.dispose()


if __name__ == "__main__":
    run_simulation()
