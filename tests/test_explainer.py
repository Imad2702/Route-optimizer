"""
Tests for natural language explanation generation.
"""

import pandas as pd
import pytest

from optimizer.explainer import explain_recommendation
from optimizer.scorer import score_routes


@pytest.fixture
def scored_sample():
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
                "risk_score": 0.80,
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
                "risk_score": 0.10,
            },
        ]
    )
    return score_routes(df, profile="balanced")


def test_explainer_output_structure(scored_sample):
    explanation = explain_recommendation(scored_sample, profile="balanced")

    assert "shipment_id" in explanation
    assert "recommended_route" in explanation
    assert "score" in explanation
    assert "headline" in explanation
    assert "narrative" in explanation
    assert "key_tradeoffs" in explanation
    assert "comparisons" in explanation

    assert isinstance(explanation["key_tradeoffs"], list)
    assert len(explanation["key_tradeoffs"]) > 0
    assert explanation["shipment_id"] == "S001"
    assert isinstance(explanation["comparisons"], dict)
    assert "vs_cheapest" in explanation["comparisons"]
    assert "vs_runner_up" in explanation["comparisons"]
    assert "It transits through" in explanation["narrative"]


def test_explainer_contains_quantified_tradeoffs(scored_sample):
    explanation = explain_recommendation(scored_sample, profile="balanced")
    tradeoff_text = " ".join(explanation["key_tradeoffs"])

    # Must contain comparisons vs alternatives
    assert ("Vs." in tradeoff_text) or ("Cost Efficiency" in tradeoff_text)
    assert len(explanation["narrative"]) > 20


def test_explainer_zero_cost_edge_case():
    # Zero freight cost must not trigger ZeroDivisionError
    df = pd.DataFrame(
        [
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "A",
                "via": "Algeciras",
                "cost_usd": 0,
                "transit_days": 30,
                "delay_probability": 0.3,
                "risk_score": 0.5,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "B",
                "via": "Singapore",
                "cost_usd": 1000,
                "transit_days": 10,
                "delay_probability": 0.1,
                "risk_score": 0.2,
            },
        ]
    )
    scored = score_routes(df, profile="time_critical")
    exp = explain_recommendation(scored, profile="time_critical")
    assert exp["recommended_route"] == "B"
    assert "vs_cheapest" in exp["comparisons"]


def test_explainer_tied_routes_handling():
    # Exactly identical routes must produce tied notes without crashing
    df = pd.DataFrame(
        [
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "A",
                "via": "Algeciras",
                "cost_usd": 2000,
                "transit_days": 20,
                "delay_probability": 0.2,
                "risk_score": 0.3,
            },
            {
                "shipment_id": "S001",
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "route_id": "B",
                "via": "Singapore",
                "cost_usd": 2000,
                "transit_days": 20,
                "delay_probability": 0.2,
                "risk_score": 0.3,
            },
        ]
    )
    scored = score_routes(df, profile="balanced")
    exp = explain_recommendation(scored, profile="balanced")
    assert exp["is_tied"] is True
    assert "Tied" in exp["headline"]
    assert any("Tie Detected" in t for t in exp["key_tradeoffs"])
