"""Business layer tests."""

from __future__ import annotations

import pandas as pd

from src.churn.business import cost_based_threshold, expected_value_table, lift_at_k, top_at_risk


def test_threshold_formula():
    # threshold = offer_cost / (success_rate * clv)
    t = cost_based_threshold(offer_cost=50, success_rate=0.5, clv=200)
    assert abs(t - 0.5) < 1e-9


def test_threshold_default_in_range():
    t = cost_based_threshold()
    assert 0 < t < 1, "threshold must be between 0 and 1"


def test_expected_value_table_sorted():
    proba = pd.Series([0.1, 0.9, 0.5], name="churn_prob")
    ev_df = expected_value_table(proba)
    # Should be sorted descending by expected_value
    assert (ev_df["expected_value"].diff().dropna() <= 0).all()


def test_ev_positive_for_high_risk():
    # A customer with 90% churn prob should have positive EV with default params
    proba = pd.Series([0.9])
    ev_df = expected_value_table(proba, offer_cost=50, success_rate=0.3, clv=500)
    assert ev_df["expected_value"].iloc[0] > 0


def test_top_at_risk_filters_threshold():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    proba = pd.Series([0.2, 0.8, 0.5, 0.9])
    result = top_at_risk(X, proba, threshold=0.6)
    # Only customers with prob >= 0.6 should appear
    assert (result["churn_prob"] >= 0.6).all()


def test_top_at_risk_sorted_descending():
    X = pd.DataFrame({"a": range(5)})
    proba = pd.Series([0.3, 0.9, 0.7, 0.8, 0.6])
    result = top_at_risk(X, proba, threshold=0.0, n=3)
    assert result["churn_prob"].iloc[0] >= result["churn_prob"].iloc[1]


def test_lift_at_k_perfect_model():
    # Perfect model: all churners ranked first → lift = 1 / base_rate * base_rate = 1/k if k >= base_rate
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # base rate = 0.2
    proba = pd.Series([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    lift = lift_at_k(y, proba, k=0.20)  # top 20% = 2 customers, both churners
    assert lift == 5.0  # precision=1.0, base=0.2 → lift=5


def test_lift_at_k_random_model():
    # Random model has lift ≈ 1
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # base rate = 0.5
    proba = pd.Series([0.5] * 10)  # uniform → random
    lift = lift_at_k(y, proba, k=0.50)
    # precision at top-5 random = base rate → lift ≈ 1
    assert 0.5 <= lift <= 2.0  # loose bound since tie-breaking is arbitrary


def test_lift_at_k_zero_base_rate():
    import math
    y = pd.Series([0, 0, 0])
    proba = pd.Series([0.9, 0.5, 0.1])
    assert math.isnan(lift_at_k(y, proba))
