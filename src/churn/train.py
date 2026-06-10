"""Train, evaluate, and persist the churn pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def train_and_evaluate(save: bool = True) -> dict:
    from src.churn.data import load_raw, split
    from src.churn.pipeline import build_pipeline

    df = load_raw()
    X_train, X_test, y_train, y_test = split(df)

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"Train: {len(y_train)} rows, {pos} churn ({pos/len(y_train)*100:.1f}%)")
    print(f"Test:  {len(y_test)} rows, {y_test.sum()} churn ({y_test.mean()*100:.1f}%)")

    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    base_rate = float(y_test.mean())

    print(f"\nTest ROC-AUC : {roc_auc:.4f}")
    print(f"Test PR-AUC  : {pr_auc:.4f}  (base rate {base_rate:.4f})")
    print(f"PR-AUC lift  : {pr_auc / base_rate:.2f}x above base rate")
    print("\n" + classification_report(y_test, (proba >= 0.5).astype(int)))

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "base_rate": round(base_rate, 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        import json
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        print(f"\nModel saved to {MODEL_PATH}")

    return {"pipeline": pipeline, "X_test": X_test, "y_test": y_test, "metrics": metrics}


if __name__ == "__main__":
    train_and_evaluate()
