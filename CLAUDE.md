# Project: Churn prediction + SHAP + FastAPI

## What this is
A churn model (sklearn Pipeline + LightGBM) with SHAP explanations, a business-action layer
(cost-based threshold + expected retention value), served via FastAPI with a Gradio demo.

## Stack
Python 3.12 · scikit-learn · LightGBM · SHAP · FastAPI/uvicorn · Gradio · joblib · pytest · ruff.

## Commands
- Train:  `python -m src.churn.train`
- API:    `uvicorn api.main:app --reload`
- Demo:   `python app.py`
- Tests:  `pytest -q`
- Lint:   `ruff check .`

## Conventions
- ALL preprocessing in one sklearn Pipeline (fit on train only, no leakage).
- Persist the whole pipeline with joblib; API loads that file at startup.
- SHAP: split pipeline — transform inputs with `pipeline[:-1]`, call
  `shap.TreeExplainer(pipeline[-1])`, map back via `get_feature_names_out()`.
- /predict returns churn_prob + top human-readable reasons (SHAP).
- Threshold chosen by business cost (not 0.5).

## Done per step
Verify command passes → commit.
