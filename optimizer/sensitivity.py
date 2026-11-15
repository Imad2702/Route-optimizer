"""
Sensitivity Analysis Module for Route Optimization.
Analyzes how route recommendations shift across different operational priority profiles.
"""

import pandas as pd

from .scorer import PROFILES, score_routes


def run_cross_profile_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates all shipments against every priority profile.

    Returns:
    --------
    pd.DataFrame:
        Table with shipment_id, origin, destination, and the recommended route_id
        for each profile, plus a stability indicator.
    """
    profiles = list(PROFILES.keys())

    # Pre-extract origin and destination map
    od_map = (
        df[["shipment_id", "origin", "destination"]]
        .drop_duplicates()
        .set_index("shipment_id")
    )

    # High-performance batch evaluation across profiles
    scored_all = {prof: score_routes(df, profile=prof) for prof in profiles}

    records = []
    for sid in df["shipment_id"].unique():
        record = {
            "shipment_id": sid,
            "origin": od_map.loc[sid, "origin"],
            "destination": od_map.loc[sid, "destination"],
        }

        recommended_routes = []
        for prof in profiles:
            s = scored_all[prof]
            winner_row = s[(s["shipment_id"] == sid) & (s["recommended"])].iloc[0]
            best_route = winner_row["route_id"]
            record[prof] = best_route
            recommended_routes.append(best_route)

        # Decision Stability: How consistent is the winner across profiles?
        unique_winners = len(set(recommended_routes))
        if unique_winners == 1:
            stability = "Very High (Robust)"
        elif unique_winners == 2:
            stability = "Moderate"
        else:
            stability = "Highly Sensitive"

        record["stability"] = stability
        records.append(record)

    return pd.DataFrame(records)


def profile_distribution_summary(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the frequency of each route type winning under each profile.
    """
    profiles = list(PROFILES.keys())
    route_types = ["A", "B", "C", "D"]

    summary_data = []
    for prof in profiles:
        counts = sensitivity_df[prof].value_counts().to_dict()
        row = {"profile": prof}
        for r in route_types:
            count = counts.get(r, 0)
            pct = round((count / len(sensitivity_df)) * 100, 1)
            row[f"Route_{r}"] = f"{count} ({pct}%)"
            row[f"Route_{r}_count"] = count
        summary_data.append(row)

    return pd.DataFrame(summary_data)


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience alias for run_cross_profile_sensitivity."""
    return run_cross_profile_sensitivity(df)
