import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    AIRLINES,
    DESTINATIONS,
    EXPECTED_COLUMNS,
    SOURCES,
    TOTAL_STOPS_MAP,
)
from src.exception import FlightFairException
from src.logger import logger


class FeatureEngineer:
    """
    Feature Engineering pipeline responsible for temporal feature extraction,
    duration standardization, total stops mapping, categorical encoding,
    and schema alignment for training and inference.
    """

    @staticmethod
    def parse_duration_string(duration_str: str) -> Tuple[int, int]:
        """
        Parses flight duration string (e.g., '2h 50m', '19h', '50m')
        into hours and minutes integers.
        """
        duration = str(duration_str).strip()
        parts = duration.split()
        
        # Standardize format so both hours and minutes exist
        if len(parts) == 1:
            if "h" in duration:
                duration = f"{duration} 0m"
            else:
                duration = f"0h {duration}"
        
        try:
            parts = duration.split()
            hours = int(parts[0].replace("h", ""))
            mins = int(parts[1].replace("m", ""))
            return hours, mins
        except Exception:
            return 0, 0

    def clean_and_engineer_dataset(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Transforms raw historical flight dataset (train or test dataframe)
        into engineered feature representation.
        """
        try:
            logger.info(f"Starting feature engineering on dataframe with shape {df.shape} (is_train={is_train})")
            data = df.copy()

            # 1. Temporal Feature Extraction - Date of Journey
            if "Date_of_Journey" in data.columns:
                journey_dates = pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y", errors="coerce")
                # Fallback for alternative date formats if any coerce to NaT
                if journey_dates.isna().any():
                    journey_dates = journey_dates.fillna(pd.to_datetime(data["Date_of_Journey"], errors="coerce"))
                data["Journey_day"] = journey_dates.dt.day.fillna(1).astype(int)
                data["Journey_month"] = journey_dates.dt.month.fillna(1).astype(int)
                data.drop("Date_of_Journey", axis=1, inplace=True, errors="ignore")

            # 2. Temporal Feature Extraction - Departure Time
            if "Dep_Time" in data.columns:
                dep_times = pd.to_datetime(data["Dep_Time"], errors="coerce")
                data["Dep_hour"] = dep_times.dt.hour.fillna(0).astype(int)
                data["Dep_min"] = dep_times.dt.minute.fillna(0).astype(int)
                data.drop("Dep_Time", axis=1, inplace=True, errors="ignore")

            # 3. Temporal Feature Extraction - Arrival Time
            if "Arrival_Time" in data.columns:
                # Arrival time may contain date suffixes like '01:10 22 Mar'
                arrival_times = pd.to_datetime(data["Arrival_Time"], errors="coerce")
                data["Arrival_hour"] = arrival_times.dt.hour.fillna(0).astype(int)
                data["Arrival_min"] = arrival_times.dt.minute.fillna(0).astype(int)
                data.drop("Arrival_Time", axis=1, inplace=True, errors="ignore")

            # 4. Standardize Flight Duration into Duration_hours and Duration_mins
            if "Duration" in data.columns:
                durations = data["Duration"].astype(str).tolist()
                parsed = [self.parse_duration_string(d) for d in durations]
                data["Duration_hours"] = [p[0] for p in parsed]
                data["Duration_mins"] = [p[1] for p in parsed]
                data.drop("Duration", axis=1, inplace=True, errors="ignore")

            # 5. Map Total_Stops to ordinal integers
            if "Total_Stops" in data.columns:
                data["Total_Stops"] = data["Total_Stops"].replace(TOTAL_STOPS_MAP)
                # Ensure missing or unmatched stops map to 0
                data["Total_Stops"] = pd.to_numeric(data["Total_Stops"], errors="coerce").fillna(0).astype(int)

            # 6. Drop unneeded columns
            drop_cols = ["Route", "Additional_Info"]
            data.drop(columns=[col for col in drop_cols if col in data.columns], axis=1, inplace=True, errors="ignore")

            # 7. One-Hot Encoding for Airline, Source, Destination
            categorical_dummies = []
            if "Airline" in data.columns:
                airline_dummies = pd.get_dummies(data["Airline"], prefix="Airline", drop_first=True, dtype=int)
                categorical_dummies.append(airline_dummies)
                data.drop("Airline", axis=1, inplace=True)

            if "Source" in data.columns:
                source_dummies = pd.get_dummies(data["Source"], prefix="Source", drop_first=True, dtype=int)
                categorical_dummies.append(source_dummies)
                data.drop("Source", axis=1, inplace=True)

            if "Destination" in data.columns:
                dest_dummies = pd.get_dummies(data["Destination"], prefix="Dest", drop_first=True, dtype=int)
                categorical_dummies.append(dest_dummies)
                data.drop("Destination", axis=1, inplace=True)

            if categorical_dummies:
                data = pd.concat([data] + categorical_dummies, axis=1)

            # Preserve target variable if present
            target = None
            if "Price" in data.columns:
                target = data["Price"]
                data.drop("Price", axis=1, inplace=True)

            # 8. Align with exact EXPECTED_COLUMNS
            for col in EXPECTED_COLUMNS:
                if col not in data.columns:
                    data[col] = 0

            # Reorder columns to guarantee exact model signature
            data = data[EXPECTED_COLUMNS]

            # Re-attach Price if it was present
            if target is not None:
                data["Price"] = target

            logger.info(f"Feature engineering completed. Output shape: {data.shape}")
            return data
        except Exception as e:
            raise FlightFairException(e, sys)

    @classmethod
    def transform_single_flight(
        cls,
        airline: str,
        source: str,
        destination: str,
        departure_datetime: datetime,
        arrival_datetime: datetime,
        stops: int
    ) -> pd.DataFrame:
        """
        Transforms a single flight input query (e.g., from Web UI or REST API)
        into a single-row DataFrame aligned with EXPECTED_COLUMNS.
        """
        try:
            # Initialize all expected columns with zero
            row_data = {col: 0 for col in EXPECTED_COLUMNS}

            # Temporal features
            row_data["Journey_day"] = departure_datetime.day
            row_data["Journey_month"] = departure_datetime.month
            row_data["Dep_hour"] = departure_datetime.hour
            row_data["Dep_min"] = departure_datetime.minute
            row_data["Arrival_hour"] = arrival_datetime.hour
            row_data["Arrival_min"] = arrival_datetime.minute

            # Compute duration
            duration_delta = arrival_datetime - departure_datetime
            total_seconds = duration_delta.total_seconds()
            if total_seconds < 0:
                # If arrival is next day without date rollover specified
                total_seconds += 24 * 3600

            row_data["Duration_hours"] = int(total_seconds // 3600)
            row_data["Duration_mins"] = int((total_seconds % 3600) // 60)
            row_data["Total_Stops"] = int(stops)

            # One-Hot Encoding matching
            airline_key = f"Airline_{airline}"
            if airline_key in row_data:
                row_data[airline_key] = 1

            source_key = f"Source_{source}"
            if source_key in row_data:
                row_data[source_key] = 1

            dest_key = f"Dest_{destination}"
            if dest_key in row_data:
                row_data[dest_key] = 1

            return pd.DataFrame([row_data], columns=EXPECTED_COLUMNS)
        except Exception as e:
            raise FlightFairException(e, sys)
