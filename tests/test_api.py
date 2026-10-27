import json
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Tests health endpoint status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_metadata_endpoint(client):
    """Tests metadata retrieval endpoint."""
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert "airlines" in data
    assert "sources" in data
    assert "destinations" in data
    assert "Delhi" in data["sources"]
    assert "Cochin" in data["destinations"]


def test_predict_endpoint_valid(client):
    """Tests prediction endpoint with valid payload."""
    payload = {
        "airline": "IndiGo",
        "source": "Delhi",
        "destination": "Cochin",
        "departure_datetime": "2026-11-20T10:30:00",
        "arrival_datetime": "2026-11-20T14:45:00",
        "stops": 1
    }

    response = client.post(
        "/api/v1/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert "predicted_price_inr" in body["data"]
    assert body["data"]["predicted_price_inr"] > 0
    assert "latency_ms" in body
    assert "route_comparisons" in body["data"]


def test_predict_endpoint_same_city(client):
    """Tests prediction endpoint returns 400 for identical origin and destination."""
    payload = {
        "airline": "IndiGo",
        "source": "Delhi",
        "destination": "Delhi",
        "departure_datetime": "2026-11-20T10:30:00",
        "arrival_datetime": "2026-11-20T14:45:00",
        "stops": 0
    }

    response = client.post(
        "/api/v1/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"


def test_predict_batch_endpoint(client):
    """Tests batch prediction endpoint."""
    payload = {
        "flights": [
            {
                "airline": "IndiGo",
                "source": "Delhi",
                "destination": "Cochin",
                "departure_datetime": "2026-11-20T10:30:00",
                "arrival_datetime": "2026-11-20T14:45:00",
                "stops": 1
            },
            {
                "airline": "SpiceJet",
                "source": "Kolkata",
                "destination": "Banglore",
                "departure_datetime": "2026-11-20T08:00:00",
                "arrival_datetime": "2026-11-20T10:30:00",
                "stops": 0
            }
        ]
    }

    response = client.post(
        "/api/v1/predict/batch",
        data=json.dumps(payload),
        content_type="application/json"
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["summary"]["total_flights"] == 2
    assert len(body["predictions"]) == 2
