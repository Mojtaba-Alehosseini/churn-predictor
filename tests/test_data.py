"""Data loading, cleaning, and split tests."""

from __future__ import annotations

from src.churn.data import SAMPLE_CSV, load_raw, split


def test_load_raw_sample():
    df = load_raw(SAMPLE_CSV)
    assert "Churn" in df.columns
    assert df["Churn"].isin([0, 1]).all(), "Churn must be binary int"
    assert "customerID" not in df.columns, "customerID must be dropped"


def test_total_charges_numeric():
    df = load_raw(SAMPLE_CSV)
    assert df["TotalCharges"].dtype in ("float64", "float32"), "TotalCharges must be numeric"


def test_split_no_leakage():
    df = load_raw(SAMPLE_CSV)
    X_train, X_test, y_train, y_test = split(df)
    overlap = X_train.index.intersection(X_test.index)
    assert len(overlap) == 0, "train and test indices must not overlap"


def test_split_stratified():
    df = load_raw(SAMPLE_CSV)
    _, _, y_train, y_test = split(df)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    # Stratification should keep churn rates within 3 percentage points
    assert abs(train_rate - test_rate) < 0.03, (
        f"churn rate mismatch: train={train_rate:.3f}, test={test_rate:.3f}"
    )
