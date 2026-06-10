"""
FastAPI app for churn prediction.

POST /predict  -> {churn_prob, top_reasons, decision}
GET  /health   -> {"status": "ok"}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "pipeline.joblib"

app = FastAPI(title="Churn Predictor API", version="1.0.0")

_pipeline = None


def _load_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python -m src.churn.train")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float | None = None

    @field_validator("SeniorCitizen")
    @classmethod
    def validate_senior(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")
        return v


class PredictionResponse(BaseModel):
    churn_prob: float
    decision: str
    top_reasons: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    pipeline = _load_model()

    row = pd.DataFrame([customer.model_dump()])
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")

    try:
        proba = float(pipeline.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from src.churn.business import cost_based_threshold
    from src.churn.explain import top_reasons

    threshold = cost_based_threshold()
    decision = "contact for retention" if proba >= threshold else "no action needed"
    reasons = top_reasons(pipeline, row, n=5)

    return PredictionResponse(churn_prob=round(proba, 4), decision=decision, top_reasons=reasons)
