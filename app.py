import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from pydantic import BaseModel, Field, field_validator

from src.config import (
    ALL_AIRLINES,
    ALL_DESTINATIONS,
    ALL_SOURCES,
    BASE_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    STATIC_DIR,
    TEMPLATES_DIR,
)
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_json

# Initialize Flask Application
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR)
)
CORS(app)

# Initialize inference pipeline
try:
    predict_pipeline = PredictPipeline(model_path=MODEL_PATH, metadata_path=MODEL_METADATA_PATH)
    logger.info("PredictPipeline initialized successfully in Flask application.")
except Exception as e:
    logger.error(f"Failed to initialize PredictPipeline: {e}")
    predict_pipeline = None


# ---------------------------------------------------------
# Pydantic Schemas for Request Validation
# ---------------------------------------------------------
class SingleFlightRequest(BaseModel):
    airline: str = Field(..., description="Airline name (e.g., 'IndiGo', 'Air India', 'SpiceJet')")
    source: str = Field(..., description="Departure airport city (e.g., 'Delhi', 'Mumbai', 'Kolkata')")
    destination: str = Field(..., description="Arrival airport city (e.g., 'Cochin', 'Banglore', 'Hyderabad')")
    departure_datetime: str = Field(..., description="Departure ISO datetime string, e.g. '2026-11-20T10:30:00'")
    arrival_datetime: str = Field(..., description="Arrival ISO datetime string, e.g. '2026-11-20T14:45:00'")
    stops: int = Field(default=0, ge=0, le=4, description="Total number of stops (0 to 4)")

    @field_validator("source", "destination")
    @classmethod
    def validate_cities(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("City name cannot be empty")
        return v.strip()


class BatchFlightRequest(BaseModel):
    flights: List[SingleFlightRequest]


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def parse_iso_datetime(dt_str: str) -> datetime:
    """Parses varied datetime string formats into datetime object."""
    dt_str = dt_str.replace("Z", "").strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime: '{dt_str}'. Supported ISO format: 'YYYY-MM-DDTHH:MM'.")


# ---------------------------------------------------------
# Web Application Route (Frontend Dashboard)
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Serves the FlightFair interactive web dashboard."""
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        metadata = load_json(MODEL_METADATA_PATH)

    return render_template(
        "index.html",
        airlines=ALL_AIRLINES,
        sources=ALL_SOURCES,
        destinations=ALL_DESTINATIONS,
        metadata=metadata
    )


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint providing service status and model diagnostics."""
    is_ready = predict_pipeline is not None and predict_pipeline.model is not None
    status_code = 200 if is_ready else 503

    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "FlightFair Predictive Pricing Engine",
        "version": "1.0.0",
        "model_loaded": is_ready,
        "framework": "Flask / XGBoost / MLflow / DVC",
        "server_time": datetime.now().isoformat()
    }), status_code


@app.route("/api/v1/metadata", methods=["GET"])
def get_metadata():
    """Returns available airlines, sources, destinations, and stop configurations."""
    return jsonify({
        "status": "success",
        "data": {
            "airlines": ALL_AIRLINES,
            "sources": ALL_SOURCES,
            "destinations": ALL_DESTINATIONS,
            "stops_options": [
                {"label": "Non-stop (Direct)", "value": 0},
                {"label": "1 Stop", "value": 1},
                {"label": "2 Stops", "value": 2},
                {"label": "3 Stops", "value": 3},
                {"label": "4 Stops", "value": 4}
            ]
        }
    }), 200


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    """
    Predicts airline ticket fare for a single flight query.
    Optionally computes route competitor pricing comparison.
    """
    start_time = time.time()

    if not predict_pipeline:
        return jsonify({
            "status": "error",
            "message": "Predictive model is not loaded. Please ensure model training is completed."
        }), 503

    try:
        req_json = request.get_json(force=True, silent=False)
        if not req_json:
            return jsonify({"status": "error", "message": "Missing JSON request body."}), 400

        # Validate with Pydantic
        flight_req = SingleFlightRequest(**req_json)

        dep_dt = parse_iso_datetime(flight_req.departure_datetime)
        arr_dt = parse_iso_datetime(flight_req.arrival_datetime)

        if flight_req.source.lower() == flight_req.destination.lower():
            return jsonify({
                "status": "error",
                "message": f"Source and Destination cannot be the same city ('{flight_req.source}')."
            }), 400

        # Execute Prediction
        prediction_result = predict_pipeline.predict_single_flight(
            airline=flight_req.airline,
            source=flight_req.source,
            destination=flight_req.destination,
            departure_datetime=dep_dt,
            arrival_datetime=arr_dt,
            stops=flight_req.stops
        )

        # Include competitor comparison on same route
        include_comparison = request.args.get("compare", "true").lower() == "true"
        comparisons = []
        if include_comparison:
            comparisons = predict_pipeline.compare_airlines_on_route(
                source=flight_req.source,
                destination=flight_req.destination,
                departure_datetime=dep_dt,
                arrival_datetime=arr_dt,
                stops=flight_req.stops
            )

        prediction_result["route_comparisons"] = comparisons

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            "status": "success",
            "data": prediction_result,
            "latency_ms": latency_ms
        }), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/v1/predict/batch", methods=["POST"])
def predict_batch():
    """Predicts fares for a batch of multiple flight queries."""
    start_time = time.time()

    if not predict_pipeline:
        return jsonify({"status": "error", "message": "Predictive model is not loaded."}), 503

    try:
        req_json = request.get_json(force=True)
        batch_req = BatchFlightRequest(**req_json)

        results = []
        for item in batch_req.flights:
            dep_dt = parse_iso_datetime(item.departure_datetime)
            arr_dt = parse_iso_datetime(item.arrival_datetime)
            res = predict_pipeline.predict_single_flight(
                airline=item.airline,
                source=item.source,
                destination=item.destination,
                departure_datetime=dep_dt,
                arrival_datetime=arr_dt,
                stops=item.stops
            )
            results.append(res)

        prices = [r["predicted_price_inr"] for r in results]
        summary = {
            "total_flights": len(results),
            "average_price_inr": round(sum(prices) / len(prices), 2) if prices else 0,
            "min_price_inr": min(prices) if prices else 0,
            "max_price_inr": max(prices) if prices else 0,
        }

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return jsonify({
            "status": "success",
            "summary": summary,
            "predictions": results,
            "latency_ms": latency_ms
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/v1/model-info", methods=["GET"])
def model_info():
    """Returns detailed training metadata, metrics, and parameters."""
    if not MODEL_METADATA_PATH.exists():
        return jsonify({
            "status": "warning",
            "message": "Metadata file not found. Model might have been loaded from default weights."
        }), 404

    try:
        data = load_json(MODEL_METADATA_PATH)
        return jsonify({
            "status": "success",
            "metadata": data
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def main():
    """Runs the Flask development server or production WSGI runner."""
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting FlightFair server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()