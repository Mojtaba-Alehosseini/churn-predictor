"""
Business-action layer: cost-based decision threshold + expected retention value.

Business model:
  - A retention offer costs `offer_cost` per customer contacted.
  - The offer is accepted (churn prevented) with probability `success_rate`.
  - A churned customer has lifetime value `clv`.
  - Expected value of calling a customer = success_rate * clv * P(churn|customer) - offer_cost
  - Optimal to call when EV > 0, i.e. P(churn) > offer_cost / (success_rate * clv).
  - This gives a natural, business-derived decision threshold.
"""

from __future__ import annotations

import pandas as pd


def cost_based_threshold(
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    clv: float = 500.0,
) -> float:
    """
    Threshold where expected retention value = 0.
    Call the customer if P(churn) > threshold.
    """
    if success_rate <= 0 or clv <= 0:
        return 0.5
    return offer_cost / (success_rate * clv)


def expected_value_table(
    proba: pd.Series,
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    clv: float = 500.0,
) -> pd.DataFrame:
    """
    Return a table with expected retention value per customer.
    EV = success_rate * clv * P(churn) - offer_cost
    """
    ev = success_rate * clv * proba - offer_cost
    df = pd.DataFrame({"churn_prob": proba, "expected_value": ev.round(2)})
    return df.sort_values("expected_value", ascending=False).reset_index(drop=True)


def top_at_risk(
    X_test: pd.DataFrame,
    proba: pd.Series,
    threshold: float,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top N at-risk customers above the decision threshold."""
    df = X_test.copy()
    df["churn_prob"] = proba.values
    df = df[df["churn_prob"] >= threshold]
    return df.nlargest(n, "churn_prob").reset_index(drop=True)


def lift_at_k(
    y_true: pd.Series,
    proba: pd.Series,
    k: float = 0.10,
) -> float:
    """
    Lift at top-k%: how much better than random is the model when contacting
    the top k fraction of customers sorted by predicted churn probability.

    lift@k = (precision in top-k) / (base rate)

    A lift of 3.0 means the top-10% list contains 3× more churners than a
    random list of the same size — directly translates to outreach ROI.
    """
    import numpy as np

    y = pd.Series(y_true).reset_index(drop=True)
    p = pd.Series(proba).reset_index(drop=True)
    n = max(1, int(np.ceil(len(y) * k)))
    top_idx = p.nlargest(n).index
    precision_at_k = y.iloc[top_idx].mean()
    base_rate = y.mean()
    if base_rate <= 0:
        return float("nan")
    return float(precision_at_k / base_rate)
