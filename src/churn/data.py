"""Load and split the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parents[2] / "data"
FULL_CSV = DATA_DIR / "telco_churn.csv"
SAMPLE_CSV = DATA_DIR / "sample_500.csv"

CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
DROP_COLS = ["customerID"]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load full data if available, else fall back to 500-row sample."""
    if path is None:
        path = FULL_CSV if FULL_CSV.exists() else SAMPLE_CSV
    df = pd.read_csv(path)
    # TotalCharges: blank strings in rows with tenure=0 → coerce to NaN (imputed later)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET].str.strip() == "Yes").astype(int)
    df = df.drop(columns=DROP_COLS)
    return df


def split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    assert X_train.index.intersection(X_test.index).empty, "leakage: train/test indices overlap"
    return X_train, X_test, y_train, y_test
