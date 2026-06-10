# Deployment guide

## FastAPI locally

```bash
uvicorn api.main:app --reload --port 8000
```

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 151.65
  }'
```

## Hugging Face Spaces (Gradio)

Requires `HF_TOKEN` in your environment:

1. Create a Space at https://huggingface.co/new-space (SDK: Gradio).
2. Push the repo; set the Space's `app_file` to `app.py`.
3. The `models/pipeline.joblib` file must be committed to load without retraining.
