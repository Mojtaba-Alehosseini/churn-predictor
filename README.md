# Customer Churn Predictor

Predict which customers are at risk of churning, **explain every prediction** with SHAP,
and surface a **business-action recommendation** (who is worth a retention offer, given
offer cost and expected lifetime value). Served via a FastAPI endpoint and a Gradio demo.

## Results (run `python -m src.churn.train` to reproduce)

Dataset: IBM Telco Customer Churn — 7,043 customers, 21 features, 26.5 % churn rate.
Train / test split: 80 / 20, stratified.

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8314** |
| **PR-AUC** | **0.6432** |
| Base-rate PR-AUC | 0.2654 |
| PR-AUC lift | **2.42x** |

PR-AUC is the headline because it measures ranking quality on the minority class
(churners) — the same population you would target with a retention budget.

**Lift@10%:** contacting the top 10% of customers ranked by predicted churn probability
reaches ~3–4× more actual churners than random outreach of the same size — directly
translating model skill into outreach ROI. `lift_at_k(y_true, proba, k=0.10)` is available
in `src.churn.business`.

### Business decision

With a retention offer costing 50, a 30 % success rate, and average CLV = 500,
the cost-based decision threshold is **0.33** (vs the arbitrary 0.5 default).
This means contacting more high-risk customers, which is optimal when the expected
retention value `(0.30 * 500 * P(churn) - 50)` is positive.

### Top SHAP drivers (global — higher = more predictive of churn)

Contract type, tenure, internet service, and monthly charges dominate.
A long-tenured customer on a two-year contract is far less likely to churn;
a new Fiber-optic customer on a month-to-month contract is high-risk.

## Approach

```
CSV (7,043 rows)
   |
   v
sklearn Pipeline:
  ColumnTransformer (impute + scale numeric, impute + one-hot categorical)
   + LightGBM classifier (scale_pos_weight handles 3.7:1 imbalance)
   |
SHAP TreeExplainer  -->  global importance + per-customer top reasons
   |
Business layer: threshold = offer_cost / (success_rate * CLV)
   |
FastAPI /predict  -->  {churn_prob, decision, top_reasons}
Gradio demo       -->  interactive web UI
```

## Run it

```bash
git clone https://github.com/Mojtaba-Alehosseini/churn-predictor
cd churn-predictor
pip install -r requirements.txt

# Download full dataset (optional — sample_500.csv committed for tests)
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
     -o data/telco_churn.csv

python -m src.churn.train      # train + save models/pipeline.joblib
uvicorn api.main:app --reload  # API at http://localhost:8000
python app.py                  # Gradio demo
pytest -q                      # tests
ruff check .                   # lint
```

### API usage

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

Response: `{"churn_prob": 0.73, "decision": "contact for retention", "top_reasons": [...]}`

## Structure

```
churn-predictor/
├── src/churn/
│   ├── data.py       # load, clean (TotalCharges), stratified split
│   ├── pipeline.py   # sklearn Pipeline: ColumnTransformer + LightGBM
│   ├── train.py      # fit, evaluate, persist pipeline.joblib
│   ├── explain.py    # SHAP global importance + per-customer reasons
│   └── business.py   # cost-based threshold + expected retention value
├── api/main.py       # FastAPI: POST /predict, GET /health
├── app.py            # Gradio demo
├── models/           # pipeline.joblib (committed after training)
├── data/             # sample_500.csv + download note
└── tests/            # data, pipeline, business, API contract tests
```

## Limitations

- Trained on the IBM Telco sample dataset. Transfer to a real telco requires
  retraining on proprietary data and recalibrating the CLV and offer-cost parameters.
- The threshold is based on a simplified expected-value model; production systems
  would incorporate uplift modelling (who responds to the offer, not just who churns).
- Gradio demo requires the trained model file; cloud deployment instructions in
  `DEPLOYMENT.md` (HF_TOKEN needed for Hugging Face Spaces).

## Attributions

- IBM Telco Customer Churn dataset — IBM Corporation (sample dataset, no OSS licence claimed)
- [scikit-learn](https://scikit-learn.org) — BSD-3-Clause
- [LightGBM](https://github.com/microsoft/LightGBM) — MIT
- [SHAP](https://github.com/shap/shap) — MIT
- [FastAPI](https://fastapi.tiangolo.com) — MIT

---

*Built by [Mojtaba Alehosseini](https://github.com/Mojtaba-Alehosseini) — data scientist.*
