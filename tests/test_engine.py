"""
Tests for high-level RouteOptimizerEngine orchestrator.
"""

import pandas as pd
import pytest

from optimizer.engine import RouteOptimizerEngine, optimize_shipment
from optimizer.models import OptimizationResult, Shipment


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


def test_engine_optimize_shipment_dataframe(sample_shipment_df):
    engine = RouteOptimizerEngine()
    res = engine.optimize_shipment(sample_shipment_df, shipment_id="S001", profile="balanced")

    assert isinstance(res, OptimizationResult)
    assert res.shipment_id == "S001"
    assert res.recommended_route is not None
    assert res.recommended_route.rank == 1
    assert res.explanation is not None
    assert "headline" in res.explanation
    assert len(res.all_routes) == 3


def test_engine_optimize_shipment_from_entity(sample_shipment_df):
    shipment = Shipment.from_dataframe(sample_shipment_df, "S001")
    res = optimize_shipment(shipment, profile="time_critical")

    assert isinstance(res, OptimizationResult)
    # Time critical should select Route C (10 days)
    assert res.recommended_route.route_id == "C"


def test_engine_compare_algorithms(sample_shipment_df):
    engine = RouteOptimizerEngine()
    comp = engine.compare_algorithms(sample_shipment_df, shipment_id="S001", profile="balanced")

    assert "wsm" in comp
    assert "topsis" in comp
    assert comp["wsm"].algorithm == "wsm"
    assert comp["topsis"].algorithm == "topsis"


def test_engine_validation_on_missing_column():
    bad_df = pd.DataFrame([{"shipment_id": "S001", "cost_usd": 1000}])
    engine = RouteOptimizerEngine()
    with pytest.raises(ValueError, match="missing required columns"):
        engine.optimize_shipment(bad_df, shipment_id="S001")
