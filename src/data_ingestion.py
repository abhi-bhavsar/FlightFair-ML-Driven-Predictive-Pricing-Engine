import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TEST_RAW_PATH, TRAIN_RAW_PATH
from src.exception import FlightFairException
from src.logger import logger


class DataIngestion:
    """
    Data Ingestion component responsible for loading raw datasets,
    performing initial schema validation, handling missing records,
    and generating train/validation partitions.
    """
    def __init__(self, train_path: Path = TRAIN_RAW_PATH, test_path: Path = TEST_RAW_PATH):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test Excel datasets."""
        try:
            logger.info(f"Loading raw train dataset from: {self.train_path}")
            if not self.train_path.exists():
                raise FileNotFoundError(f"Train dataset file not found at: {self.train_path}")
            train_df = pd.read_excel(self.train_path)
            logger.info(f"Train dataset loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

            logger.info(f"Loading raw test dataset from: {self.test_path}")
            if not self.test_path.exists():
                raise FileNotFoundError(f"Test dataset file not found at: {self.test_path}")
            test_df = pd.read_excel(self.test_path)
            logger.info(f"Test dataset loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

            # Handle missing values (in train, Route and Total_Stops have 1 null record)
            initial_train_len = len(train_df)
            train_df.dropna(subset=["Route", "Total_Stops"], inplace=True)
            if len(train_df) < initial_train_len:
                logger.info(f"Dropped {initial_train_len - len(train_df)} record(s) with missing Route/Total_Stops")

            return train_df, test_df
        except Exception as e:
            raise FlightFairException(e, sys)

    def split_train_val(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits training dataframe into train and validation splits."""
        try:
            logger.info(f"Splitting training data with test_size={test_size}, random_state={random_state}")
            train_split, val_split = train_test_split(df, test_size=test_size, random_state=random_state)
            logger.info(f"Train split size: {len(train_split)} | Val split size: {len(val_split)}")
            return train_split, val_split
        except Exception as e:
            raise FlightFairException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_df, test_df = ingestion.load_data()
    print("Ingestion test passed successfully!")
