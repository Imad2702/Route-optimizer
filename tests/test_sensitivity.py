"""
Tests for cross-profile sensitivity and stability analysis.
"""

import pytest

from data.generate_data import generate_shipments
from optimizer.sensitivity import (
    profile_distribution_summary,
    run_cross_profile_sensitivity,
)


@pytest.fixture(scope="module")
def generated_df():
    return generate_shipments()


def test_sensitivity_covers_all_shipments(generated_df):
    sens_df = run_cross_profile_sensitivity(generated_df)

    assert len(sens_df) == 30
    assert "shipment_id" in sens_df.columns
    assert "balanced" in sens_df.columns
    assert "cost_sensitive" in sens_df.columns
    assert "time_critical" in sens_df.columns
    assert "risk_averse" in sens_df.columns
    assert "stability" in sens_df.columns

    # Stability should only be one of the defined categories
    valid_stabilities = {"Very High (Robust)", "Moderate", "Highly Sensitive"}
    assert set(sens_df["stability"].unique()).issubset(valid_stabilities)


def test_profile_distribution_summary(generated_df):
    sens_df = run_cross_profile_sensitivity(generated_df)
    summary = profile_distribution_summary(sens_df)

    assert len(summary) == 4  # 4 profiles
    assert "profile" in summary.columns
    for r in ["A", "B", "C", "D"]:
        assert f"Route_{r}_count" in summary.columns

    # For each profile, sum of route counts must equal 30
    for _, row in summary.iterrows():
        total = sum(row[f"Route_{r}_count"] for r in ["A", "B", "C", "D"])
        assert total == 30
