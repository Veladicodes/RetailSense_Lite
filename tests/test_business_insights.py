"""Unit tests for the pure business-logic helpers in utils/business_insights.py.

They use small synthetic frames, so they need no dataset, Streamlit or Prophet.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.business_insights import (  # noqa: E402
    calculate_price_elasticity,
    detect_sales_anomalies,
    optimize_price,
)


def weekly_frame(sales, prices=None, product="Milk"):
    weeks = pd.date_range("2024-01-01", periods=len(sales), freq="W-MON")
    frame = pd.DataFrame({"week_start": weeks, "product_name": product, "sales_qty": sales})
    if prices is not None:
        frame["price"] = prices
    return frame


def test_optimize_price_elastic_demand_returns_consistent_numbers():
    result = optimize_price(current_price=100.0, elasticity=-2.0, margin=0.3)
    assert set(result) == {
        "current_price",
        "optimal_price",
        "price_change_pct",
        "expected_quantity_change_pct",
        "profit_gain_pct",
        "profit_gain_abs",
    }
    cost = 100.0 * (1 - 0.3)
    assert result["optimal_price"] >= round(cost * 1.1, 2)


def test_optimize_price_inelastic_demand_raises_price_ten_percent():
    result = optimize_price(current_price=100.0, elasticity=-0.4, margin=0.3)
    assert result["optimal_price"] == pytest.approx(110.0)
    assert result["price_change_pct"] == pytest.approx(10.0)
    assert result["expected_quantity_change_pct"] < 0


def test_elasticity_falls_back_to_default_with_too_little_data():
    frame = weekly_frame([100] * 6, prices=[10.0] * 6)
    assert calculate_price_elasticity(frame, "Milk") == -1.2


def test_elasticity_recovers_a_negative_slope_from_a_constant_elasticity_curve():
    prices = np.array([10.0, 10.5, 11.0, 10.2, 9.8, 10.8, 11.2, 9.6, 10.4, 11.4, 10.0, 9.9, 10.7, 11.1, 10.3])
    sales = 1000 * prices ** -1.5
    frame = weekly_frame(sales, prices=prices)
    elasticity = calculate_price_elasticity(frame, "Milk")
    assert elasticity == pytest.approx(-1.5, abs=0.05)


def test_anomalies_empty_for_short_history():
    frame = weekly_frame([100] * 5)
    result = detect_sales_anomalies(frame, "Milk")
    assert result.empty
    assert "severity" in result.columns


def test_anomalies_flag_an_injected_spike():
    sales = [100, 102, 98, 101, 99, 103, 97, 100, 102, 98, 101, 99, 100, 400, 101, 99, 100, 102, 98, 100]
    frame = weekly_frame(sales)
    result = detect_sales_anomalies(frame, "Milk")
    assert not result.empty
    spike_week = frame["week_start"].iloc[13]
    assert spike_week in set(result["date"])
    assert result.loc[result["date"] == spike_week, "severity"].iloc[0] == "Severe"
