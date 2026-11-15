"""
Synthetic dataset generator for shipment route optimization.
Generates:
1. data/shipments.csv (30 shipments × 4 routes = 120 route alternatives)
2. data/shipment_legs.csv (240 individual multi-leg records: 2 legs per route)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Origin-Destination pairs (10 unique pairs, each repeated 3×)
# ---------------------------------------------------------------------------
OD_PAIRS = [
    ("Shanghai", "Rotterdam"),
    ("Mumbai", "Dubai"),
    ("Shenzhen", "Los Angeles"),
    ("Singapore", "Hamburg"),
    ("Hong Kong", "New York"),
    ("Busan", "Long Beach"),
    ("Jakarta", "Antwerp"),
    ("Chennai", "Felixstowe"),
    ("Guangzhou", "Vancouver"),
    ("Bangkok", "Le Havre"),
]

# Hub choices per route type
HUBS = {
    "A": ["Colombo", "Port Said", "Algeciras", "Tanjung Pelepas", "Klang"],
    "B": ["Singapore", "Dubai", "Suez", "Malta", "Bremerhaven"],
    "C": ["Jebel Ali", "Port Klang", "Colombo", "Piraeus", "Gioia Tauro"],
    "D": ["Salalah", "Colon", "Kingston", "Las Palmas", "Dakar"],
}


def _route_params(route_id: str) -> dict:
    """
    Return base cost/time/risk parameters by route type.

    Route A: cheapest, highest risk & delay
    Route B: balanced middle
    Route C: fastest, most expensive
    Route D: medium cost, medium risk
    """
    if route_id == "A":
        cost_range = (1500, 3000)
        time_range = (25, 45)
        delay_range = (0.40, 0.65)
        congestion_range = (0.55, 1.0)
        geo_range = (0.50, 1.0)
    elif route_id == "B":
        cost_range = (2500, 4500)
        time_range = (15, 30)
        delay_range = (0.20, 0.45)
        congestion_range = (0.30, 0.65)
        geo_range = (0.25, 0.60)
    elif route_id == "C":
        cost_range = (5000, 8000)
        time_range = (5, 18)
        delay_range = (0.05, 0.22)
        congestion_range = (0.05, 0.35)
        geo_range = (0.05, 0.30)
    else:  # D
        cost_range = (3000, 5500)
        time_range = (12, 28)
        delay_range = (0.15, 0.40)
        congestion_range = (0.20, 0.55)
        geo_range = (0.20, 0.55)
    return {
        "cost_range": cost_range,
        "time_range": time_range,
        "delay_range": delay_range,
        "congestion_range": congestion_range,
        "geo_range": geo_range,
    }


def generate_shipments() -> pd.DataFrame:
    """Generate 120 route alternatives across 30 shipments."""
    rows = []
    shipment_num = 1

    # 10 OD pairs × 3 repetitions = 30 shipments
    for rep in range(3):
        for origin, destination in OD_PAIRS:
            shipment_id = f"S{shipment_num:03d}"

            for route_id in ["A", "B", "C", "D"]:
                p = _route_params(route_id)

                cost_usd = int(np.random.uniform(*p["cost_range"]))
                transit_days = int(np.random.uniform(*p["time_range"]))
                delay_probability = round(np.random.uniform(*p["delay_range"]), 2)
                congestion_risk = round(np.random.uniform(*p["congestion_range"]), 2)
                geopolitical_risk = round(np.random.uniform(*p["geo_range"]), 2)
                risk_score = round(0.5 * congestion_risk + 0.5 * geopolitical_risk, 2)

                via = np.random.choice(HUBS[route_id])

                rows.append(
                    {
                        "shipment_id": shipment_id,
                        "origin": origin,
                        "destination": destination,
                        "route_id": route_id,
                        "via": via,
                        "cost_usd": cost_usd,
                        "transit_days": transit_days,
                        "delay_probability": delay_probability,
                        "congestion_risk": congestion_risk,
                        "geopolitical_risk": geopolitical_risk,
                        "risk_score": risk_score,
                    }
                )

            shipment_num += 1

    return pd.DataFrame(rows)


def generate_shipment_legs(shipments_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Decompose routes into distinct operational legs (Origin -> Hub, Hub -> Destination).
    Returns 240 leg records (2 legs per route option).
    """
    if shipments_df is None:
        shipments_df = generate_shipments()

    leg_rows = []
    for _, r in shipments_df.iterrows():
        sid = r["shipment_id"]
        rid = r["route_id"]
        orig = r["origin"]
        dest = r["destination"]
        via = r["via"]
        tot_c = float(r["cost_usd"])
        tot_t = float(r["transit_days"])
        tot_dp = float(r["delay_probability"])
        tot_cr = float(r["congestion_risk"])
        tot_gr = float(r["geopolitical_risk"])

        # Leg 1: Mainline Ocean Trunk (Origin -> Hub)
        c1 = round(tot_c * 0.58, 2)
        t1 = round(tot_t * 0.55, 1)
        dp1 = round(tot_dp * 0.55, 2)
        cr1 = round(tot_cr * 0.85, 2)
        gr1 = round(tot_gr * 0.90, 2)
        rs1 = round(0.5 * cr1 + 0.5 * gr1, 2)

        leg_rows.append(
            {
                "shipment_id": sid,
                "route_id": rid,
                "leg_index": 1,
                "from_node": orig,
                "to_node": via,
                "transport_mode": "Mainline Ocean Trunk",
                "cost_usd": c1,
                "transit_days": t1,
                "delay_probability": dp1,
                "congestion_risk": cr1,
                "geopolitical_risk": gr1,
                "risk_score": rs1,
            }
        )

        # Leg 2: Secondary Feeder / Final Approach (Hub -> Destination)
        c2 = round(tot_c - c1, 2)
        t2 = round(tot_t - t1, 1)
        dp2 = round(tot_dp * 0.45, 2)
        cr2 = round(tot_cr * 1.15, 2)
        gr2 = round(tot_gr * 1.10, 2)
        rs2 = round(0.5 * cr2 + 0.5 * gr2, 2)

        leg_rows.append(
            {
                "shipment_id": sid,
                "route_id": rid,
                "leg_index": 2,
                "from_node": via,
                "to_node": dest,
                "transport_mode": "Regional Feeder / Intermodal",
                "cost_usd": c2,
                "transit_days": t2,
                "delay_probability": dp2,
                "congestion_risk": cr2,
                "geopolitical_risk": gr2,
                "risk_score": rs2,
            }
        )

    return pd.DataFrame(leg_rows)


def main():
    df = generate_shipments()
    legs_df = generate_shipment_legs(df)

    # Save to data/ relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "shipments.csv")
    legs_out_path = os.path.join(script_dir, "shipment_legs.csv")

    df.to_csv(out_path, index=False)
    legs_df.to_csv(legs_out_path, index=False)

    print(f"Generated {len(df)} routes → {out_path}")
    print(f"Generated {len(legs_df)} operational legs → {legs_out_path}")
    print(
        f"Shipments: {df['shipment_id'].nunique()}, Routes per shipment: {df['route_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
