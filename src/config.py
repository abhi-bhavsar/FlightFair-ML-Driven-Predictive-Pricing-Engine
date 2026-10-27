import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "model"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure essential directories exist
for directory in [DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_RAW_PATH = DATA_RAW_DIR / "train_data.xlsx"
TEST_RAW_PATH = DATA_RAW_DIR / "test_data.xlsx"
TRAIN_PROCESSED_PATH = DATA_PROCESSED_DIR / "train_clean.csv"
TEST_PROCESSED_PATH = DATA_PROCESSED_DIR / "test_clean.csv"
MODEL_PATH = MODEL_DIR / "best_xgboost_flight_model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# MLflow Configuration
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}".replace("\\", "/"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "FlightFair_Predictive_Pricing")

# Schema & Feature Definitions
AIRLINES = [
    "Air India",
    "GoAir",
    "IndiGo",
    "Jet Airways",
    "Jet Airways Business",
    "Multiple carriers",
    "Multiple carriers Premium economy",
    "SpiceJet",
    "Trujet",
    "Vistara",
    "Vistara Premium economy"
]

# Note: Air Asia is the reference baseline omitted by drop_first=True
ALL_AIRLINES = ["Air Asia"] + AIRLINES

SOURCES = [
    "Chennai",
    "Delhi",
    "Kolkata",
    "Mumbai"
]

# Note: Banglore is the reference baseline omitted by drop_first=True
ALL_SOURCES = ["Banglore"] + SOURCES

DESTINATIONS = [
    "Cochin",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "New Delhi"
]

# Note: Banglore is the reference baseline omitted by drop_first=True
ALL_DESTINATIONS = ["Banglore"] + DESTINATIONS

TOTAL_STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}

# The exact 29 features required by the trained XGBoost model
EXPECTED_COLUMNS = [
    'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 
    'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins', 
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways', 
    'Airline_Jet Airways Business', 'Airline_Multiple carriers', 
    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet', 'Airline_Trujet', 
    'Airline_Vistara', 'Airline_Vistara Premium economy', 'Source_Chennai', 
    'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Dest_Cochin', 
    'Dest_Delhi', 'Dest_Hyderabad', 'Dest_Kolkata', 'Dest_New Delhi'
]

# Hyperparameter search space for tuning
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 7, 9, 13],
    "learning_rate": [0.05, 0.1, 0.2, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "random_state": [42]
}

# Best known baseline parameters from exploration
XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42
}
