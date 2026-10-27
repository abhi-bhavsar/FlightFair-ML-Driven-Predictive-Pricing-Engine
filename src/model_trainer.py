import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.config import (
    EXPECTED_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    XGB_DEFAULT_PARAMS,
    XGB_PARAM_DISTRIBUTIONS,
)
from src.exception import FlightFairException
from src.logger import logger
from src.utils import (
    evaluate_regression_metrics,
    plot_actual_vs_predicted,
    plot_feature_importances,
    plot_residuals,
    save_json,
    save_object,
)


class ModelTrainer:
    """
    Model Trainer component responsible for training regression models,
    hyperparameter tuning with RandomizedSearchCV, tracking experiments
    with MLflow, generating evaluation plots, and serializing the model.
    """

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        model_save_path: Path = MODEL_PATH,
        metadata_save_path: Path = MODEL_METADATA_PATH
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model_save_path = model_save_path
        self.metadata_save_path = metadata_save_path

        # Setup MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow configured - Tracking URI: {self.tracking_uri} | Experiment: {self.experiment_name}")

    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 15,
        cv: int = 3,
        random_state: int = 42
    ) -> Tuple[XGBRegressor, Dict[str, Any]]:
        """
        Executes hyperparameter tuning for XGBoost Regressor using RandomizedSearchCV.
        """
        try:
            logger.info(f"Starting XGBoost RandomizedSearchCV tuning (n_iter={n_iter}, cv={cv})...")
            base_model = XGBRegressor(random_state=random_state, n_jobs=-1)

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=XGB_PARAM_DISTRIBUTIONS,
                n_iter=n_iter,
                cv=cv,
                scoring="r2",
                verbose=1,
                random_state=random_state,
                n_jobs=-1
            )
            search.fit(X_train, y_train)

            logger.info(f"Best XGBoost Hyperparameters: {search.best_params_}")
            logger.info(f"Best Cross-Validation R2 Score: {search.best_score_:.4f}")
            return search.best_estimator_, search.best_params_
        except Exception as e:
            raise FlightFairException(e, sys)

    def train_and_track(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune: bool = False,
        run_name: Optional[str] = None
    ) -> Tuple[XGBRegressor, Dict[str, float]]:
        """
        Trains the XGBoost model, evaluates metrics on validation data,
        logs parameters, metrics, plots, and serialized model to MLflow.
        """
        if run_name is None:
            run_name = f"XGBoost_{'Tuned' if tune else 'Production'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Initiating MLflow run: '{run_name}'")

        try:
            with mlflow.start_run(run_name=run_name) as run:
                start_time = time.time()

                if tune:
                    model, best_params = self.tune_xgboost(X_train, y_train)
                else:
                    logger.info("Training XGBoost with tuned default production parameters...")
                    best_params = XGB_DEFAULT_PARAMS
                    model = XGBRegressor(**best_params)
                    model.fit(X_train, y_train)

                training_duration = round(time.time() - start_time, 2)
                logger.info(f"Model training finished in {training_duration} seconds.")

                # Evaluation on validation set
                y_pred = model.predict(X_val)
                metrics = evaluate_regression_metrics(y_val.to_numpy(), y_pred)
                logger.info(f"Validation Metrics: {metrics}")

                # MLflow Logging: Parameters & Tags
                mlflow.log_params(best_params)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("features_count", len(EXPECTED_COLUMNS))
                mlflow.log_param("training_duration_sec", training_duration)
                mlflow.set_tag("model_family", "XGBoost")
                mlflow.set_tag("pipeline", "FlightFair Predictive Pricing Engine")

                # MLflow Logging: Metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Generate and log diagnostic visual artifacts
                plots_dir = Path("reports/figures")
                plots_dir.mkdir(parents=True, exist_ok=True)

                feat_imp_path = plots_dir / "feature_importances.png"
                act_pred_path = plots_dir / "actual_vs_predicted.png"
                residuals_path = plots_dir / "residuals_distribution.png"

                plot_feature_importances(model, EXPECTED_COLUMNS, feat_imp_path)
                plot_actual_vs_predicted(y_val.to_numpy(), y_pred, act_pred_path)
                plot_residuals(y_val.to_numpy(), y_pred, residuals_path)

                mlflow.log_artifact(str(feat_imp_path), artifact_path="plots")
                mlflow.log_artifact(str(act_pred_path), artifact_path="plots")
                mlflow.log_artifact(str(residuals_path), artifact_path="plots")

                # Save serialized model object
                save_object(self.model_save_path, model)

                # Save metadata JSON
                metadata = {
                    "model_type": "XGBRegressor",
                    "training_timestamp": datetime.now().isoformat(),
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id,
                    "hyperparameters": best_params,
                    "metrics": metrics,
                    "features": EXPECTED_COLUMNS,
                    "training_duration_sec": training_duration,
                }
                save_json(self.metadata_save_path, metadata)
                mlflow.log_artifact(str(self.metadata_save_path), artifact_path="metadata")

                # Log model into MLflow Model Registry artifact store using native XGBoost flavor
                try:
                    mlflow.xgboost.log_model(
                        xgb_model=model,
                        artifact_path="model",
                        input_example=X_val.head(2)
                    )
                except Exception as log_err:
                    logger.warning(f"Native xgboost log_model fallback: {log_err}")
                    mlflow.log_artifact(str(self.model_save_path), artifact_path="model")

                logger.info(f"Model and MLflow artifacts logged successfully! Run ID: {run.info.run_id}")
                return model, metrics
        except Exception as e:
            raise FlightFairException(e, sys)
