import os
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn
from utils.mlflow_config import setup_mlflow, MLFLOW_MODEL_NAME
from utils.files_util import save_files, load_files
import utils.ml_pipeline_config as config


def experiment():

    x_train, x_test, y_train, y_test = load_files(['x_train', 'x_test', 'y_train', 'y_test'])

    # the maximum number of principal components to investigate cannot be higher than the number of coolumns in the dataset
    max_pca_components = config.params["max_pca_components"] if config.params["max_pca_components"] <= x_train.shape[1] else x_train.shape[1]
    cv_folds = config.params["cv_folds"]
    logreg_maxiter = config.params["logreg_maxiter"]

    # --- Configurar MLflow ---
    setup_mlflow()

    with mlflow.start_run() as run:
        # Registrar parametros del experimento
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("logreg_maxiter", logreg_maxiter)
        mlflow.log_param("max_pca_components", max_pca_components)
        mlflow.log_param("test_split_ratio", config.params["test_split_ratio"])

        # pipeline definition
        std_scaler = StandardScaler()
        pca = PCA(max_pca_components-1)
        log_reg = LogisticRegression(max_iter=logreg_maxiter)

        pipe = Pipeline(steps=[('std_scaler', std_scaler),
                            ('pca', pca),
                            ('log_reg', log_reg)])

        # parameters for hyper-parameter tuning
        params = {
            'pca__n_components': list(range(1, max_pca_components)),
            'log_reg__C': np.logspace(0.05, 0.1, 1)
        }

        # cross-validated training through grid search
        grid_search = GridSearchCV(pipe, params, cv=cv_folds)
        grid_search.fit(x_train, y_train)

        # selection of the best parameters
        best_c = round(grid_search.best_params_.get("log_reg__C"),2)
        best_princ_comp = grid_search.best_params_.get("pca__n_components")

        # performances on test set
        y_test_predicted = grid_search.best_estimator_.predict(x_test)
        test_set_accuracy = round(accuracy_score(y_test, y_test_predicted),3)

        # Registrar metricas en MLflow
        mlflow.log_metric("test_set_accuracy", test_set_accuracy)
        mlflow.log_metric("best_cv_score", round(grid_search.best_score_, 3))
        mlflow.log_param("best_logreg_c", best_c)
        mlflow.log_param("best_pca_components", best_princ_comp)

        # Registrar artefactos: confusion matrix y classification report
        cm = confusion_matrix(y_test, y_test_predicted)
        cr = classification_report(y_test, y_test_predicted)

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.csv")
            pd.DataFrame(cm).to_csv(cm_path, index=False)
            mlflow.log_artifact(cm_path)

            cr_path = os.path.join(tmpdir, "classification_report.txt")
            with open(cr_path, "w") as f:
                f.write(cr)
            mlflow.log_artifact(cr_path)

        # Registrar modelo en MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        # save experiments information for historical persistence
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        exp_info = pd.DataFrame([[now,
                              cv_folds,
                              logreg_maxiter,
                              max_pca_components,
                              best_c,
                              best_princ_comp,
                              test_set_accuracy,
                              run.info.run_id]],
                              columns=['experiment_datetime',
                                       'cv_folds',
                                       'logreg_maxiter',
                                       'max_pca_components',
                                       'best_logreg_c',
                                       'best_pca_components',
                                       'test_set_accuracy',
                                       'mlflow_run_id',
                                       ])
        exp_info.name = 'exp_info'

        save_files([exp_info])
