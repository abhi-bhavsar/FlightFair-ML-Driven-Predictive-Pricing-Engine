import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import FlightFairException
from src.logger import logger


def save_object(file_path: Path | str, obj: Any) -> None:
    """Serializes a Python object to disk using pickle."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Serialized object saved successfully to: {path}")
    except Exception as e:
        raise FlightFairException(e, sys)


def load_object(file_path: Path | str) -> Any:
    """Loads a serialized Python object from disk."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded successfully from: {path}")
        return obj
    except Exception as e:
        raise FlightFairException(e, sys)


def save_json(file_path: Path | str, data: Dict[str, Any]) -> None:
    """Saves dictionary data to a formatted JSON file."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON data saved to: {path}")
    except Exception as e:
        raise FlightFairException(e, sys)


def load_json(file_path: Path | str) -> Dict[str, Any]:
    """Loads data from a JSON file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise FlightFairException(e, sys)


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes key regression metrics: R2, MAE, MSE, RMSE, and MAPE.
    """
    try:
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        
        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        metrics = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2)
        }
        return metrics
    except Exception as e:
        raise FlightFairException(e, sys)


def plot_feature_importances(model: Any, feature_names: list[str], save_path: Path | str) -> str:
    """Plots and saves the top feature importances of the tree-based model."""
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return ""

        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]

        plt.figure(figsize=(10, 6), dpi=150)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        palette = sns.color_palette("Blues_r", len(top_features))
        
        ax = sns.barplot(x=top_importances, y=top_features, palette=palette)
        plt.title("FlightFair: Top 15 Feature Importances (XGBoost)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Engineered Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Feature importance plot saved to: {path}")
        return str(path)
    except Exception as e:
        raise FlightFairException(e, sys)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | str) -> str:
    """Plots Actual vs Predicted ticket prices scatter plot."""
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6), dpi=150)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color="#1e40af")
        
        # 45-degree reference line
        max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
        plt.plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal Fit (y = x)")
        
        plt.title("Actual vs Predicted Flight Ticket Prices (₹)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Actual Price (₹)", fontsize=11)
        plt.ylabel("Predicted Price (₹)", fontsize=11)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Actual vs Predicted plot saved to: {path}")
        return str(path)
    except Exception as e:
        raise FlightFairException(e, sys)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path | str) -> str:
    """Plots distribution of prediction residuals."""
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        residuals = y_true - y_pred

        plt.figure(figsize=(8, 5), dpi=150)
        sns.histplot(residuals, kde=True, color="#059669", bins=50)
        plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
        plt.title("Residual Distribution (Actual - Predicted)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Residual (₹)", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        logger.info(f"Residual distribution plot saved to: {path}")
        return str(path)
    except Exception as e:
        raise FlightFairException(e, sys)
