"""API contract tests: valid payload, invalid payload, health endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

VALID_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 151.65,
}

MODEL_PATH = ROOT / "models" / "pipeline.joblib"


@pytest.fixture(scope="module")
def client():
    """Return a FastAPI TestClient only if the model exists."""
    if not MODEL_PATH.exists():
        pytest.skip("model not trained yet — run python -m src.churn.train")

    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_prob" in body
    assert 0.0 <= body["churn_prob"] <= 1.0
    assert "top_reasons" in body
    assert len(body["top_reasons"]) > 0
    assert "decision" in body


def test_predict_invalid_senior(client):
    bad = {**VALID_CUSTOMER, "SeniorCitizen": 5}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # pydantic validation error


def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
