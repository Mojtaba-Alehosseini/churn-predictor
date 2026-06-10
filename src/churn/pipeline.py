"""Build the sklearn Pipeline: ColumnTransformer + LightGBM classifier."""

from __future__ import annotations

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn.data import CAT_COLS, NUM_COLS


def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline.

    preprocessing:
        numeric  -> median impute -> standard scale
        categorical -> most-frequent impute -> one-hot (handle_unknown='ignore')
    estimator:
        LightGBMClassifier with scale_pos_weight to handle class imbalance
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ])
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
