"""
Tests for scoring and multi-criteria utility calculations.
"""

import pandas as pd
import pytest

from optimizer.scorer import (
    PROFILES,
    get_profile_weights,
    normalize_criterion,
    recommend_route,
    score_routes,
)


@pytest.fixture
def sample_shipment_df():
    """Create a minimal synthetic shipment with 4 contrasting routes."""
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
                "risk_score": 0.80,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "B",
                "via": "Singapore",
                "cost_usd": 3500,
                "transit_days": 25,
                "delay_probability": 0.30,
                "risk_score": 0.40,
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
                "risk_score": 0.10,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "D",
                "via": "Las Palmas",
                "cost_usd": 4500,
                "transit_days": 18,
                "delay_probability": 0.20,
                "risk_score": 0.30,
            },
        ]
    )


def test_profile_weights_sum_to_one():
    for name in PROFILES:
        w = get_profile_weights(name)
        assert pytest.approx(sum(w.values()), 0.001) == 1.0


def test_invalid_profile_raises_error():
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile_weights("non_existent_profile")


def test_custom_weights_normalized():
    custom = {"cost": 2.0, "time": 1.0, "risk": 1.0}
    w = get_profile_weights(custom_weights=custom)
    assert w["cost"] == 0.5
    assert w["time"] == 0.25
    assert w["risk"] == 0.25
    assert sum(w.values()) == 1.0


def test_negative_weight_raises_error():
    custom = {"cost": -1.0, "time": 2.0, "risk": 1.0}
    with pytest.raises(ValueError, match="cannot be negative"):
        get_profile_weights(custom_weights=custom)


def test_non_numeric_weight_raises_error():
    custom = {"cost": "expensive", "time": 1.0, "risk": 1.0}
    with pytest.raises(ValueError, match="must be numeric"):
        get_profile_weights(custom_weights=custom)


def test_schema_validation_null_values(sample_shipment_df):
    bad_df = sample_shipment_df.copy()
    bad_df.loc[0, "cost_usd"] = None
    with pytest.raises(ValueError, match="contains null/NaN values"):
        score_routes(bad_df)


def test_schema_validation_negative_values(sample_shipment_df):
    bad_df = sample_shipment_df.copy()
    bad_df.loc[0, "cost_usd"] = -100
    with pytest.raises(ValueError, match="cannot be negative"):
        score_routes(bad_df)


def test_schema_validation_invalid_risk(sample_shipment_df):
    bad_df = sample_shipment_df.copy()
    bad_df.loc[0, "risk_score"] = 1.5
    with pytest.raises(ValueError, match="valid range"):
        score_routes(bad_df)


def test_tied_routes_detected_in_scorer(sample_shipment_df):
    tied_df = sample_shipment_df.copy()
    # Give route A and B identical metrics
    tied_df.loc[1, "cost_usd"] = tied_df.loc[0, "cost_usd"]
    tied_df.loc[1, "transit_days"] = tied_df.loc[0, "transit_days"]
    tied_df.loc[1, "risk_score"] = tied_df.loc[0, "risk_score"]

    scored = score_routes(tied_df, profile="cost_sensitive")
    assert "is_tied" in scored.columns
    # Both top routes should be flagged as tied
    assert scored["is_tied"].any()


def test_unsupported_algorithm_raises_error(sample_shipment_df):
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        score_routes(sample_shipment_df, algorithm="genetic_algo")


def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError, match="is empty"):
        score_routes(pd.DataFrame())


def test_normalize_criterion_inverted():
    # Lower value must produce higher utility (1.0 = best)
    series = pd.Series([100, 200, 300])
    norm = normalize_criterion(series, lower_is_better=True)
    assert norm.iloc[0] == 1.0  # lowest cost gets 1.0
    assert norm.iloc[1] == 0.5
    assert norm.iloc[2] == 0.0  # highest cost gets 0.0


def test_normalize_criterion_constant_series():
    series = pd.Series([50, 50, 50])
    norm = normalize_criterion(series, lower_is_better=True)
    assert (norm == 1.0).all()


def test_score_routes_structure_and_bounds(sample_shipment_df):
    scored = score_routes(sample_shipment_df, profile="balanced")

    assert "composite_score" in scored.columns
    assert "rank" in scored.columns
    assert "recommended" in scored.columns

    # Scores must be within [0, 100]
    assert (scored["composite_score"] >= 0).all()
    assert (scored["composite_score"] <= 100).all()

    # Exactly one recommended winner
    recommended = scored[scored["recommended"]]
    assert len(recommended) == 1
    assert recommended.iloc[0]["rank"] == 1


def test_profile_shift_behavior(sample_shipment_df):
    # Under cost-sensitive, the cheapest route (A) or balanced (B) should outperform the $7000 express route (C)
    cost_scored = score_routes(sample_shipment_df, profile="cost_sensitive")
    c_cost_score = cost_scored[cost_scored["route_id"] == "C"].iloc[0][
        "composite_score"
    ]
    a_cost_score = cost_scored[cost_scored["route_id"] == "A"].iloc[0][
        "composite_score"
    ]
    assert a_cost_score > c_cost_score

    # Under time-critical, the fastest route (C, 10 days) must beat the 40-day route (A)
    time_scored = score_routes(sample_shipment_df, profile="time_critical")
    c_time_score = time_scored[time_scored["route_id"] == "C"].iloc[0][
        "composite_score"
    ]
    a_time_score = time_scored[time_scored["route_id"] == "A"].iloc[0][
        "composite_score"
    ]
    assert c_time_score > a_time_score
    assert time_scored[time_scored["recommended"]].iloc[0]["route_id"] == "C"


def test_recommend_route_helper(sample_shipment_df):
    scored = score_routes(sample_shipment_df, profile="balanced")
    rec = recommend_route(scored, shipment_id="S001")
    assert len(rec) == 1
    assert bool(rec.iloc[0]["recommended"]) is True
