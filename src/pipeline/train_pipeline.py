import argparse
import sys
from pathlib import Path

from src.config import TRAIN_PROCESSED_PATH
from src.data_ingestion import DataIngestion
from src.exception import FlightFairException
from src.feature_engineering import FeatureEngineer
from src.logger import logger
from src.model_trainer import ModelTrainer


class TrainPipeline:
    """
    End-to-end Machine Learning training pipeline orchestrator.
    Handles data ingestion, feature engineering, train/val split,
    model training, hyperparameter tuning, and MLflow experiment tracking.
    """

    def __init__(self, tune: bool = False):
        self.tune = tune
        self.ingestion = DataIngestion()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()

    def run(self):
        """Executes the full pipeline workflow."""
        try:
            logger.info("=" * 60)
            logger.info("STARTING FLIGHTFAIR ML TRAINING PIPELINE")
            logger.info("=" * 60)

            # Step 1: Data Ingestion
            logger.info("Stage 1: Ingesting raw flight datasets...")
            train_raw, _ = self.ingestion.load_data()

            # Step 2: Feature Engineering
            logger.info("Stage 2: Applying temporal feature engineering and one-hot encoding...")
            engineered_df = self.feature_engineer.clean_and_engineer_dataset(train_raw, is_train=True)

            # Save processed data for DVC pipeline reproducibility
            TRAIN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
            engineered_df.to_csv(TRAIN_PROCESSED_PATH, index=False)
            logger.info(f"Processed dataset saved to: {TRAIN_PROCESSED_PATH}")

            # Step 3: Train / Validation Split
            logger.info("Stage 3: Splitting into train and validation sets...")
            X = engineered_df.drop("Price", axis=1)
            y = engineered_df["Price"]

            train_df, val_df = self.ingestion.split_train_val(engineered_df, test_size=0.2, random_state=42)
            X_train = train_df.drop("Price", axis=1)
            y_train = train_df["Price"]
            X_val = val_df.drop("Price", axis=1)
            y_val = val_df["Price"]

            # Step 4: Model Training, Tuning & MLflow Experiment Tracking
            logger.info(f"Stage 4: Training XGBoost Regressor (Hyperparameter Tuning={self.tune})...")
            model, metrics = self.model_trainer.train_and_track(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                tune=self.tune
            )

            logger.info("=" * 60)
            logger.info("FLIGHTFAIR TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Final Validation R2 Score: {metrics['r2_score']}")
            logger.info(f"Final Validation MAE: INR {metrics['mae']}")
            logger.info(f"Final Validation RMSE: INR {metrics['rmse']}")
            logger.info("=" * 60)

            return metrics
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise FlightFairException(e, sys)


def main():
    parser = argparse.ArgumentParser(description="FlightFair ML Training Pipeline")
    parser.add_argument(
        "--tune", 
        action="store_true", 
        help="Run hyperparameter tuning with RandomizedSearchCV"
    )
    args = parser.parse_args()

    pipeline = TrainPipeline(tune=args.tune)
    pipeline.run()


if __name__ == "__main__":
    main()
