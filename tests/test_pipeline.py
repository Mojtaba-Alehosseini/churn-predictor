"""Pipeline fit/transform/predict tests."""

from __future__ import annotations

import numpy as np

from src.churn.data import SAMPLE_CSV, load_raw, split
from src.churn.pipeline import build_pipeline


def _fitted_pipeline():
    df = load_raw(SAMPLE_CSV)
    X_train, X_test, y_train, y_test = split(df)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def test_pipeline_fits_and_predicts():
    pipe, X_test, _ = _fitted_pipeline()
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2), "should return two-class probabilities"
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "proba rows must sum to 1"


def test_no_nan_in_predictions():
    pipe, X_test, _ = _fitted_pipeline()
    proba = pipe.predict_proba(X_test)
    assert not np.isnan(proba).any(), "no NaN in predictions"


def test_model_reload(tmp_path):
    import joblib

    pipe, X_test, _ = _fitted_pipeline()
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    orig_proba = pipe.predict_proba(X_test)
    loaded_proba = loaded.predict_proba(X_test)
    assert np.allclose(orig_proba, loaded_proba), "reloaded model must give same predictions"
