from datetime import datetime
import pytest

from src.pipeline.predict_pipeline import PredictPipeline


@pytest.fixture(scope="module")
def pipeline():
    return PredictPipeline()


def test_pipeline_model_loaded(pipeline):
    """Verifies that the serialized XGBoost model is loaded."""
    assert pipeline.model is not None


def test_predict_single_flight(pipeline):
    """Tests single flight price prediction calculation and business metrics."""
    dep = datetime(2026, 11, 20, 9, 30)
    arr = datetime(2026, 11, 20, 13, 15)

    result = pipeline.predict_single_flight(
        airline="Air India",
        source="Delhi",
        destination="Cochin",
        departure_datetime=dep,
        arrival_datetime=arr,
        stops=1
    )

    assert "predicted_price_inr" in result
    assert "predicted_price_usd" in result
    assert result["predicted_price_inr"] > 1000
    assert result["predicted_price_usd"] > 10
    assert result["duration_formatted"] == "3h 45m"
    assert result["price_tier"] in [
        "Budget Economy", "Standard Economy", 
        "Premium / Peak Demand", "Business / High Demand"
    ]
    assert result["confidence_interval"]["min_inr"] <= result["predicted_price_inr"]
    assert result["confidence_interval"]["max_inr"] >= result["predicted_price_inr"]


def test_identical_source_destination_error(pipeline):
    """Verifies that identical origin and destination raises ValueError."""
    dep = datetime(2026, 11, 20, 9, 30)
    arr = datetime(2026, 11, 20, 13, 15)

    with pytest.raises(Exception):
        pipeline.predict_single_flight(
            airline="IndiGo",
            source="Delhi",
            destination="Delhi",
            departure_datetime=dep,
            arrival_datetime=arr,
            stops=0
        )


def test_compare_airlines_on_route(pipeline):
    """Verifies multi-airline comparison functionality."""
    dep = datetime(2026, 11, 20, 9, 30)
    arr = datetime(2026, 11, 20, 13, 15)

    comparisons = pipeline.compare_airlines_on_route(
        source="Delhi",
        destination="Cochin",
        departure_datetime=dep,
        arrival_datetime=arr,
        stops=1
    )

    assert isinstance(comparisons, list)
    assert len(comparisons) > 3
    # Check sorted ascending by price
    prices = [c["price_inr"] for c in comparisons]
    assert prices == sorted(prices)
