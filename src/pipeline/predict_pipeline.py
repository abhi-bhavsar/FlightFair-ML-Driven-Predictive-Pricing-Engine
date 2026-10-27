import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALL_AIRLINES,
    EXPECTED_COLUMNS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)
from src.exception import FlightFairException
from src.feature_engineering import FeatureEngineer
from src.logger import logger
from src.utils import load_json, load_object


class PredictPipeline:
    """
    Production Inference Pipeline for FlightFair.
    Handles serialized model loading, feature transformation,
    single and batch flight price predictions, and route fare comparisons.
    """

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata = None
        self._load_model_and_metadata()

    def _load_model_and_metadata(self):
        """Loads serialized XGBoost model and training metadata."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Trained model not found at: {self.model_path}")
            self.model = load_object(self.model_path)
            logger.info("Trained XGBoost model loaded successfully into PredictPipeline.")

            if self.metadata_path.exists():
                self.metadata = load_json(self.metadata_path)
        except Exception as e:
            raise FlightFairException(e, sys)

    def predict_dataframe(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Executes raw predictions on a DataFrame containing the expected 29 features.
        """
        try:
            # Ensure exact column presence and ordering
            missing_cols = [c for c in EXPECTED_COLUMNS if c not in features_df.columns]
            if missing_cols:
                for col in missing_cols:
                    features_df[col] = 0
            aligned_df = features_df[EXPECTED_COLUMNS]

            predictions = self.model.predict(aligned_df)
            # Clip negative predictions to minimum reasonable fare
            predictions = np.clip(predictions, a_min=1500, a_max=None)
            return predictions
        except Exception as e:
            raise FlightFairException(e, sys)

    def predict_single_flight(
        self,
        airline: str,
        source: str,
        destination: str,
        departure_datetime: datetime,
        arrival_datetime: datetime,
        stops: int
    ) -> Dict[str, Any]:
        """
        Predicts ticket fare for a single flight query and generates
        contextual pricing metrics and intelligence.
        """
        try:
            # Input validation
            if source.strip().lower() == destination.strip().lower():
                raise ValueError("Source airport and Destination airport cannot be identical.")

            # Calculate total duration
            delta = arrival_datetime - departure_datetime
            total_seconds = delta.total_seconds()
            if total_seconds <= 0:
                # Flight lands next day
                total_seconds += 24 * 3600
                arrival_datetime = departure_datetime + timedelta(seconds=total_seconds)

            duration_hours = int(total_seconds // 3600)
            duration_mins = int((total_seconds % 3600) // 60)

            # Feature transformation
            features_df = FeatureEngineer.transform_single_flight(
                airline=airline,
                source=source,
                destination=destination,
                departure_datetime=departure_datetime,
                arrival_datetime=arrival_datetime,
                stops=stops
            )

            # Predict
            pred_fare = float(self.predict_dataframe(features_df)[0])
            pred_fare_rounded = round(pred_fare, 2)

            # Convert to USD (approx exchange rate 1 USD ~ 86.5 INR)
            usd_rate = 86.5
            pred_fare_usd = round(pred_fare_rounded / usd_rate, 2)

            # Categorize Price Range
            if pred_fare_rounded < 5000:
                tier = "Budget Economy"
                badge = "success"
                advice = "Great value flight! Fares in this bracket are lower than average."
            elif pred_fare_rounded < 10000:
                tier = "Standard Economy"
                badge = "primary"
                advice = "Competitive standard market fare for this route."
            elif pred_fare_rounded < 16000:
                tier = "Premium / Peak Demand"
                badge = "warning"
                advice = "High demand or peak travel period. Booking early is recommended."
            else:
                tier = "Business / High Demand"
                badge = "danger"
                advice = "Premium tier flight or last-minute surge pricing."

            # Confidence bounds (+/- 10% or MAE)
            lower_bound = max(1500, round(pred_fare_rounded * 0.90, 2))
            upper_bound = round(pred_fare_rounded * 1.10, 2)

            result = {
                "airline": airline,
                "source": source,
                "destination": destination,
                "departure_time": departure_datetime.strftime("%Y-%m-%d %H:%M"),
                "arrival_time": arrival_datetime.strftime("%Y-%m-%d %H:%M"),
                "total_stops": stops,
                "duration_formatted": f"{duration_hours}h {duration_mins}m",
                "duration_hours": duration_hours,
                "duration_mins": duration_mins,
                "predicted_price_inr": pred_fare_rounded,
                "predicted_price_usd": pred_fare_usd,
                "price_tier": tier,
                "badge_type": badge,
                "booking_advice": advice,
                "confidence_interval": {
                    "min_inr": lower_bound,
                    "max_inr": upper_bound
                }
            }
            return result
        except Exception as e:
            raise FlightFairException(e, sys)

    def compare_airlines_on_route(
        self,
        source: str,
        destination: str,
        departure_datetime: datetime,
        arrival_datetime: datetime,
        stops: int
    ) -> List[Dict[str, Any]]:
        """
        Calculates predicted fares across all supported airlines for the same route and timing.
        """
        comparisons = []
        for airline in ALL_AIRLINES:
            try:
                res = self.predict_single_flight(
                    airline=airline,
                    source=source,
                    destination=destination,
                    departure_datetime=departure_datetime,
                    arrival_datetime=arrival_datetime,
                    stops=stops
                )
                comparisons.append({
                    "airline": airline,
                    "price_inr": res["predicted_price_inr"],
                    "price_usd": res["predicted_price_usd"],
                    "tier": res["price_tier"]
                })
            except Exception:
                continue

        # Sort by price ascending
        comparisons.sort(key=lambda x: x["price_inr"])
        return comparisons
