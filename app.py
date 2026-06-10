"""
Gradio demo for the churn predictor.
Usage: python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "pipeline.joblib"


def predict(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
):
    if not MODEL_PATH.exists():
        return "Model not found. Run: python -m src.churn.train", ""

    pipeline = joblib.load(MODEL_PATH)

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior_citizen),
        "Partner": partner, "Dependents": dependents, "tenure": int(tenure),
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges), "TotalCharges": float(total_charges or 0),
    }])

    proba = float(pipeline.predict_proba(row)[0, 1])

    from src.churn.business import cost_based_threshold
    from src.churn.explain import top_reasons

    threshold = cost_based_threshold()
    decision = "CONTACT for retention offer" if proba >= threshold else "No action needed"
    reasons = top_reasons(pipeline, row, n=5)
    reasons_text = "\n".join(
        f"  {r['feature']} = {r['value']:.2f}  ({r['direction']}, SHAP={r['shap_value']:+.3f})"
        for r in reasons
    )

    result = f"Churn probability: {proba:.1%}\nDecision (threshold={threshold:.2f}): {decision}"
    return result, f"Top reasons:\n{reasons_text}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Checkbox(label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Slider(0, 72, step=1, label="Tenure (months)", value=12),
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method", value="Electronic check",
        ),
        gr.Number(label="Monthly Charges", value=65.0),
        gr.Number(label="Total Charges", value=780.0),
    ],
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="SHAP Reasons")],
    title="Customer Churn Predictor",
    description="Predict churn probability and see SHAP-based reasons. Decision threshold set by offer cost/CLV.",
)

if __name__ == "__main__":
    demo.launch()
