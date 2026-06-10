"""Business layer tests."""

from __future__ import annotations

import pandas as pd

from src.churn.business import cost_based_threshold, expected_value_table


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
