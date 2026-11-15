"""
Scoring and Multi-Criteria Decision Framework for Route Optimization.
Supports fully-vectorized Weighted Sum Model (WSM) and TOPSIS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Defined priority profiles with criterion weights
PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "cost": 0.33,
        "time": 0.33,
        "risk": 0.34,
        "description": "Equal balance across freight cost, transit speed, and operational risk.",
    },
    "cost_sensitive": {
        "cost": 0.60,
        "time": 0.20,
        "risk": 0.20,
        "description": "Prioritizes budget economy over transit duration and delay volatility.",
    },
    "time_critical": {
        "cost": 0.20,
        "time": 0.60,
        "risk": 0.20,
        "description": "Prioritizes urgent transit delivery times, accepting higher freight premiums.",
    },
    "risk_averse": {
        "cost": 0.20,
        "time": 0.20,
        "risk": 0.60,
        "description": "Avoids chokepoint delays, port congestions, and geopolitical vulnerabilities.",
    },
}

SUPPORTED_ALGORITHMS = ("wsm", "topsis")
REQUIRED_COLUMNS = {"shipment_id", "route_id", "cost_usd", "transit_days", "risk_score"}


def validate_shipment_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame has necessary columns, non-null values, and valid ranges for optimization."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")

    # Check for NaN / null values in required optimization columns
    null_cols = [c for c in REQUIRED_COLUMNS if df[c].isnull().any()]
    if null_cols:
        raise ValueError(f"Input DataFrame contains null/NaN values in columns: {sorted(null_cols)}")

    # Check for non-negative values in cost and transit days
    if (df["cost_usd"] < 0).any():
        raise ValueError("Freight cost ('cost_usd') cannot be negative.")
    if (df["transit_days"] < 0).any():
        raise ValueError("Transit duration ('transit_days') cannot be negative.")
    if (df["risk_score"] < 0).any() or (df["risk_score"] > 1.0).any():
        raise ValueError("Risk score ('risk_score') must be within the valid range [0.0, 1.0].")


def get_profile_weights(
    profile: str = "balanced", custom_weights: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Retrieve and validate weights for cost, time, and risk.
    Guarantees weights are non-negative, numeric, cover required keys, and normalized to sum to 1.0.
    """
    if custom_weights is not None:
        if not isinstance(custom_weights, dict):
            raise TypeError("custom_weights must be a dictionary.")
        criteria = ["cost", "time", "risk"]
        weights: dict[str, float] = {}
        for c in criteria:
            val = custom_weights.get(c, 0.0)
            try:
                fval = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Weight for '{c}' must be numeric (got {val}).")
            if fval < 0:
                raise ValueError(f"Criterion weight for '{c}' cannot be negative (got {fval}).")
            weights[c] = fval
    else:
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: {list(PROFILES.keys())}"
            )
        weights = {
            k: float(v)
            for k, v in PROFILES[profile].items()
            if k in ["cost", "time", "risk"]
        }

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sum of criteria weights must be strictly positive.")

    # Normalize weights so they sum to 1.0
    return {k: v / total for k, v in weights.items()}


def normalize_criterion(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """
    Min-Max normalize a series into [0, 1] utility where 1.0 is the best and 0.0 is the worst.
    """
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(1.0, index=series.index)

    if lower_is_better:
        # Lower raw value -> higher utility score (1.0 = lowest raw cost/time/risk)
        return (max_val - series) / (max_val - min_val)
    else:
        return (series - min_val) / (max_val - min_val)


def _compute_vectorized_utilities(res: pd.DataFrame) -> pd.DataFrame:
    """Compute relative Min-Max utilities vectorized per shipment group."""
    criteria = [
        ("cost_usd", "util_cost"),
        ("transit_days", "util_time"),
        ("risk_score", "util_risk"),
    ]
    for col, ucol in criteria:
        max_v = res.groupby("shipment_id")[col].transform("max")
        min_v = res.groupby("shipment_id")[col].transform("min")
        diff = max_v - min_v
        res[ucol] = np.where(diff > 0, (max_v - res[col]) / diff, 1.0)
    return res


def _score_routes_vectorized_wsm(
    res: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    """Vectorized calculation of Inverted Min-Max Weighted Sum Model (WSM)."""
    res = _compute_vectorized_utilities(res)
    raw = (
        weights["cost"] * res["util_cost"]
        + weights["time"] * res["util_time"]
        + weights["risk"] * res["util_risk"]
    )
    res["composite_score"] = (raw * 100).round(2)
    return res


def _score_routes_vectorized_topsis(
    res: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    """
    Vectorized calculation of TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    Computes Euclidean distances to positive-ideal and negative-ideal solutions without iterative slicing.
    """
    # Traceability utilities
    res = _compute_vectorized_utilities(res)

    criteria = [
        ("cost_usd", "cost"),
        ("transit_days", "time"),
        ("risk_score", "risk"),
    ]

    d_pos_sq = np.zeros(len(res))
    d_neg_sq = np.zeros(len(res))

    for col, wkey in criteria:
        # Group sum of squares for vector normalization
        sum_sq = res.groupby("shipment_id")[col].transform(lambda s: np.sum(s**2))
        denom = np.sqrt(sum_sq)
        denom = np.where(denom == 0, 1.0, denom)
        v = (res[col] / denom) * weights[wkey]

        # Cost criteria: ideal best is min, ideal worst is max
        best = v.groupby(res["shipment_id"]).transform("min")
        worst = v.groupby(res["shipment_id"]).transform("max")

        d_pos_sq += (v - best) ** 2
        d_neg_sq += (v - worst) ** 2

    d_pos = np.sqrt(d_pos_sq)
    d_neg = np.sqrt(d_neg_sq)
    total_dist = d_pos + d_neg
    closeness = np.where(total_dist > 0, d_neg / total_dist, 1.0)

    res["composite_score"] = np.round(closeness * 100, 2)
    return res


def score_routes(
    df: pd.DataFrame,
    profile: str = "balanced",
    custom_weights: dict[str, float] | None = None,
    algorithm: str = "wsm",
) -> pd.DataFrame:
    """
    Evaluates and scores routes across shipment groups using vectorized MCDM optimization.
    Eliminates iterative groupby loops for maximum performance and vector efficiency.

    Parameters:
    -----------
    df : pd.DataFrame
        Shipment data with columns: shipment_id, route_id, cost_usd, transit_days, risk_score.
    profile : str
        Profile name from PROFILES ('balanced', 'cost_sensitive', 'time_critical', 'risk_averse').
    custom_weights : Optional[Dict[str, float]]
        Custom weights dict with keys 'cost', 'time', 'risk'.
    algorithm : str
        Optimization algorithm: 'wsm' (Weighted Sum Model) or 'topsis'.

    Returns:
    --------
    pd.DataFrame:
        Enhanced dataframe containing utility components, composite_score (0-100),
        rank (1 = best), recommended (boolean), and algorithm tag.
    """
    validate_shipment_schema(df)
    algo_key = algorithm.lower().strip()
    if algo_key not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Choose from: {list(SUPPORTED_ALGORITHMS)}"
        )

    weights = get_profile_weights(profile=profile, custom_weights=custom_weights)
    res = df.copy()

    # Fast vectorized scoring (no manual python for-loop over groups)
    if algo_key == "topsis":
        res = _score_routes_vectorized_topsis(res, weights)
    else:
        res = _score_routes_vectorized_wsm(res, weights)

    # Vectorized ranking within each shipment
    res["rank"] = (
        res.groupby("shipment_id")["composite_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    # Track whether top recommendation is tied
    res["is_tied"] = (
        res.groupby("shipment_id")["rank"].transform(lambda r: (r == 1).sum()) > 1
    )
    res["recommended"] = res["rank"] == 1
    res["evaluated_profile"] = profile
    res["algorithm"] = algo_key

    return res


def recommend_route(
    scored_df: pd.DataFrame, shipment_id: str | None = None
) -> pd.DataFrame:
    """
    Extract the top recommended route(s).
    If shipment_id is provided, returns the recommendation for that shipment.
    """
    if shipment_id:
        subset = scored_df[scored_df["shipment_id"] == shipment_id]
        return subset[subset["recommended"]].copy()
    return scored_df[scored_df["recommended"]].copy()
