"""
Tests for TOPSIS MCDM algorithm.
"""

import pandas as pd
import pytest

from optimizer.scorer import score_routes


@pytest.fixture
def sample_shipment_df():
    return pd.DataFrame(
        [
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "A",
                "via": "Algeciras",
                "cost_usd": 2000,
                "transit_days": 40,
                "delay_probability": 0.50,
                "congestion_risk": 0.70,
                "geopolitical_risk": 0.70,
                "risk_score": 0.70,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "B",
                "via": "Singapore",
                "cost_usd": 3500,
                "transit_days": 20,
                "delay_probability": 0.25,
                "congestion_risk": 0.35,
                "geopolitical_risk": 0.35,
                "risk_score": 0.35,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "C",
                "via": "Piraeus",
                "cost_usd": 7000,
                "transit_days": 10,
                "delay_probability": 0.10,
                "congestion_risk": 0.10,
                "geopolitical_risk": 0.10,
                "risk_score": 0.10,
            },
        ]
    )


def test_topsis_scoring_structure(sample_shipment_df):
    scored = score_routes(sample_shipment_df, profile="balanced", algorithm="topsis")

    assert "composite_score" in scored.columns
    assert "rank" in scored.columns
    assert "recommended" in scored.columns
    assert (scored["algorithm"] == "topsis").all()

    # Scores must be bounded within [0, 100]
    assert (scored["composite_score"] >= 0).all()
    assert (scored["composite_score"] <= 100).all()

    # Exactly one winner
    assert scored["recommended"].sum() == 1
    assert scored[scored["recommended"]]["rank"].iloc[0] == 1


def test_topsis_cost_sensitive_favors_economy(sample_shipment_df):
    scored = score_routes(sample_shipment_df, profile="cost_sensitive", algorithm="topsis")
    winner = scored[scored["recommended"]].iloc[0]
    # In cost sensitive, Route C ($7000) should definitely not win
    assert winner["route_id"] in ["A", "B"]
