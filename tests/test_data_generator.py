"""
Tests for synthetic data generation reproducibility and integrity.
"""

from data.generate_data import OD_PAIRS, generate_shipments


def test_data_dimensions_and_schema():
    df = generate_shipments()

    assert len(df) == 120  # 30 shipments × 4 routes
    assert df["shipment_id"].nunique() == 30
    assert df["route_id"].nunique() == 4

    expected_cols = [
        "shipment_id",
        "origin",
        "destination",
        "route_id",
        "via",
        "cost_usd",
        "transit_days",
        "delay_probability",
        "congestion_risk",
        "geopolitical_risk",
        "risk_score",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Cost, transit days, risks should be positive
    assert (df["cost_usd"] > 0).all()
    assert (df["transit_days"] > 0).all()
    assert (df["delay_probability"] >= 0).all()
    assert (df["risk_score"] >= 0).all()


def test_origin_destination_pairs_covered():
    df = generate_shipments()
    pairs = set(zip(df["origin"], df["destination"]))
    assert len(pairs) == len(OD_PAIRS)
