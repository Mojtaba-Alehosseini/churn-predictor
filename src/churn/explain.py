"""
SHAP explanations for the churn pipeline.

Gotcha: shap.TreeExplainer must receive the *raw model* (pipeline[-1]),
not the full pipeline. We transform inputs with pipeline[:-1] first, then
recover original feature names via get_feature_names_out().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _clean_feature_name(name: str) -> str:
    """Remove sklearn's column-transformer prefix (e.g. 'num__tenure' -> 'tenure')."""
    return name.split("__", 1)[-1] if "__" in name else name


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline[:-1]
    raw = preprocessor.get_feature_names_out()
    return [_clean_feature_name(n) for n in raw]


def make_explainer(pipeline: Pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(pipeline[-1])


def shap_values_for_rows(
    pipeline: Pipeline,
    X: pd.DataFrame,
    explainer: shap.TreeExplainer | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Return (shap_matrix, feature_names) for the churn class (class=1).
    shap_matrix shape: (n_rows, n_features).
    """
    preprocessor = pipeline[:-1]
    X_transformed = preprocessor.transform(X)
    feature_names = get_feature_names(pipeline)

    if explainer is None:
        explainer = make_explainer(pipeline)

    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        # LightGBM binary: sv is a list [class0, class1]
        sv = sv[1]
    return sv, feature_names


def global_importance(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| across all rows — global feature importance."""
    sv, feature_names = shap_values_for_rows(pipeline, X)
    return pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)


def top_reasons(
    pipeline: Pipeline,
    row: pd.DataFrame,
    n: int = 5,
    explainer: shap.TreeExplainer | None = None,
) -> list[dict]:
    """
    Return the top N SHAP-based reasons for a single prediction.
    Positive SHAP = pushes toward churn; negative = pushes against.
    Each reason: {feature, value, shap_value, direction}.
    """
    sv, feature_names = shap_values_for_rows(pipeline, row, explainer=explainer)
    shap_row = sv[0]
    feature_values = pipeline[:-1].transform(row)[0]

    indices = np.argsort(np.abs(shap_row))[::-1][:n]
    reasons = []
    for i in indices:
        reasons.append({
            "feature": feature_names[i],
            "value": float(feature_values[i]),
            "shap_value": float(shap_row[i]),
            "direction": "increases risk" if shap_row[i] > 0 else "decreases risk",
        })
    return reasons
