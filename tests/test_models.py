"""
Unit tests for domain entities and data models (Leg, Route, ScoredRoute, Shipment, OptimizationResult).
"""

import pandas as pd

from optimizer.models import (
    Leg,
    Route,
    Shipment,
)


def test_leg_initialization_and_risk():
    leg = Leg(
        leg_index=1,
        origin="Shanghai",
        destination="Singapore",
        transport_mode="Ocean Trunk",
        cost_usd=1500.0,
        transit_days=10.0,
        delay_probability=0.20,
        congestion_risk=0.40,
        geopolitical_risk=0.20,
    )
    assert leg.leg_index == 1
    assert leg.risk_score == 0.30
    assert leg.to_dict()["cost_usd"] == 1500.0
    assert leg.to_dict()["risk_score"] == 0.30


def test_route_from_row_with_synthetic_legs():
    row = {
        "shipment_id": "S001",
        "origin": "Shanghai",
        "destination": "Rotterdam",
        "route_id": "B",
        "via": "Singapore",
        "cost_usd": 3000,
        "transit_days": 20,
        "delay_probability": 0.25,
        "congestion_risk": 0.30,
        "geopolitical_risk": 0.40,
        "risk_score": 0.35,
    }
    route = Route.from_row(row)
    assert route.route_id == "B"
    assert route.num_legs == 2
    assert len(route.legs) == 2
    assert route.legs[0].origin == "Shanghai"
    assert route.legs[0].destination == "Singapore"
    assert route.legs[1].origin == "Singapore"
    assert route.legs[1].destination == "Rotterdam"


def test_shipment_to_from_dataframe():
    df = pd.DataFrame(
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
                "geopolitical_risk": 0.60,
                "risk_score": 0.65,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "B",
                "via": "Singapore",
                "cost_usd": 3500,
                "transit_days": 20,
                "delay_probability": 0.20,
                "congestion_risk": 0.30,
                "geopolitical_risk": 0.30,
                "risk_score": 0.30,
            },
        ]
    )
    shipment = Shipment.from_dataframe(df, "S001")
    assert shipment.shipment_id == "S001"
    assert len(shipment.routes) == 2

    re_df = shipment.to_dataframe()
    assert len(re_df) == 2
    assert "cost_usd" in re_df.columns
