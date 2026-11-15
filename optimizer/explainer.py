"""
Natural Language Explanation Generator for Route Recommendations.
Provides operator-friendly trade-off reasoning, pairwise comparisons, and audit-ready justifications.
Robust against zero-division edge cases and handles score ties gracefully.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def explain_recommendation(
    scored_group: pd.DataFrame,
    profile: str = "balanced",
) -> dict[str, Any]:
    """
    Generate an interpretable explanation for the recommended route within a single shipment.

    Parameters:
    -----------
    scored_group : pd.DataFrame
        Scored dataframe containing all candidate routes for one shipment_id.
    profile : str
        Evaluation profile used ('balanced', 'cost_sensitive', 'time_critical', 'risk_averse').

    Returns:
    --------
    Dict[str, Any] containing:
        - 'shipment_id': str
        - 'recommended_route': str
        - 'score': float
        - 'profile': str
        - 'headline': str
        - 'narrative': str
        - 'key_tradeoffs': List[str]
        - 'comparisons': Dict of pairwise differences vs runner_up, cheapest, fastest, safest
        - 'is_tied': bool
        - 'tied_routes': List[str]
    """
    if "rank" not in scored_group.columns or "composite_score" not in scored_group.columns:
        raise ValueError("Input dataframe must contain 'rank' and 'composite_score'.")

    # Sort by rank, then secondary deterministic tie-breakers: lower risk, lower cost, route_id
    sorted_routes = scored_group.sort_values(
        by=["rank", "risk_score", "cost_usd", "route_id"]
    ).reset_index(drop=True)
    winner = sorted_routes.iloc[0]
    runner_up = sorted_routes.iloc[1] if len(sorted_routes) > 1 else None

    # Reference benchmark routes
    cheapest = scored_group.loc[scored_group["cost_usd"].idxmin()]
    fastest = scored_group.loc[scored_group["transit_days"].idxmin()]
    safest = scored_group.loc[scored_group["risk_score"].idxmin()]

    w_id = str(winner["route_id"])
    w_cost = float(winner["cost_usd"])
    w_time = float(winner["transit_days"])
    w_risk = float(winner["risk_score"])
    w_delay = float(winner.get("delay_probability", 0.0))
    w_via = str(winner.get("via", "Direct"))
    w_score = float(winner["composite_score"])
    algo = str(winner.get("algorithm", "wsm")).upper()

    tradeoffs: list[str] = []
    comparisons: dict[str, Any] = {}

    # Check for ties among #1 ranked routes
    tied_candidates = sorted_routes[sorted_routes["rank"] == 1]["route_id"].astype(str).tolist()
    is_tied = len(tied_candidates) > 1
    other_tied = [r for r in tied_candidates if r != w_id]

    if is_tied:
        tradeoffs.append(
            f"Ranking Note (Tie Detected): Route {w_id} is tied in composite score ({w_score:.1f}/100) "
            f"with Route(s) {', '.join(other_tied)}. Deterministic tie-breaker selected Route {w_id} "
            f"due to lower operational risk index ({w_risk:.2f})."
        )

    # Comparison vs cheapest
    c_cost = float(cheapest["cost_usd"])
    c_time = float(cheapest["transit_days"])
    c_risk = float(cheapest["risk_score"])

    if winner["route_id"] != cheapest["route_id"]:
        cost_diff = w_cost - c_cost
        cost_pct = round((cost_diff / c_cost) * 100, 1) if c_cost > 0 else 0.0
        time_saved = c_time - w_time
        time_pct = round((time_saved / c_time) * 100, 1) if c_time > 0 else 0.0
        risk_diff = round(c_risk - w_risk, 2)
        risk_pct = round((risk_diff / c_risk) * 100, 1) if c_risk > 0 else 0.0

        cost_phrase = f"${int(cost_diff):+,d} (+{cost_pct}%)"
        if time_saved > 0:
            time_phrase = f"{int(time_saved)} days faster (-{time_pct}%)"
        elif time_saved < 0:
            time_phrase = f"{int(-time_saved)} days slower (+{-time_pct}%)"
        else:
            time_phrase = "identical transit duration"

        if risk_pct > 0:
            risk_phrase = f"{risk_pct}% lower risk"
        elif risk_pct < 0:
            risk_phrase = f"{-risk_pct}% higher risk"
        else:
            risk_phrase = "identical risk profile"

        tradeoffs.append(
            f"Vs. Cheapest Option (Route {cheapest['route_id']}): "
            f"Incurs a freight premium of {cost_phrase}, but delivers {time_phrase} and operates with {risk_phrase}."
        )
        comparisons["vs_cheapest"] = {
            "route_id": str(cheapest["route_id"]),
            "cost_delta": cost_diff,
            "cost_pct": cost_pct,
            "time_saved_days": time_saved,
            "risk_reduction_pct": risk_pct,
        }
    else:
        tradeoffs.append(
            f"Cost Efficiency: Route {w_id} is already the most economical choice at ${int(w_cost):,d}."
        )
        comparisons["vs_cheapest"] = {
            "route_id": w_id,
            "is_cheapest": True,
            "cost_usd": w_cost,
        }

    # Comparison vs runner-up
    if runner_up is not None:
        r_id = str(runner_up["route_id"])
        score_gap = round(w_score - float(runner_up["composite_score"]), 2)
        cost_delta = w_cost - float(runner_up["cost_usd"])
        time_delta = w_time - float(runner_up["transit_days"])
        risk_delta = round(w_risk - float(runner_up["risk_score"]), 2)

        edge_reasons = []
        if cost_delta < 0:
            edge_reasons.append(f"${abs(int(cost_delta)):,d} lower cost")
        elif cost_delta > 0:
            edge_reasons.append(f"${int(cost_delta):,d} cost premium")

        if time_delta < 0:
            edge_reasons.append(f"{abs(int(time_delta))} days faster")
        elif time_delta > 0:
            edge_reasons.append(f"{int(time_delta)} days longer")

        if risk_delta < 0:
            edge_reasons.append(f"{abs(risk_delta):.2f} lower risk index")
        elif risk_delta > 0:
            edge_reasons.append(f"{risk_delta:.2f} higher risk index")

        if score_gap == 0:
            tradeoffs.append(
                f"Vs. Tied Alternative (Route {r_id}): Exact composite score tie ({w_score:.1f} pts). "
                f"Differentiating metrics: {', '.join(edge_reasons) if edge_reasons else 'identical criteria'}."
            )
        else:
            tradeoffs.append(
                f"Vs. Closest Alternative (Route {r_id}): Outperformed by {score_gap:.1f} score points. "
                f"Key differentiators: {', '.join(edge_reasons) if edge_reasons else 'identical criteria'}."
            )

        comparisons["vs_runner_up"] = {
            "route_id": r_id,
            "score_gap": score_gap,
            "cost_delta": cost_delta,
            "time_delta": time_delta,
            "risk_delta": risk_delta,
            "is_tie": score_gap == 0,
        }

    # Record fastest and safest reference in comparisons
    comparisons["fastest_route"] = {
        "route_id": str(fastest["route_id"]),
        "transit_days": float(fastest["transit_days"]),
    }
    comparisons["safest_route"] = {
        "route_id": str(safest["route_id"]),
        "risk_score": float(safest["risk_score"]),
    }

    # Synthesis by profile
    profile_descriptions = {
        "balanced": f"balances reasonable transit duration ({int(w_time)} days) with an acceptable risk score of {w_risk:.2f} without incurring unnecessary cost overruns",
        "cost_sensitive": f"achieves superior cost containment at ${int(w_cost):,d} while meeting tolerable threshold boundaries for time and risk",
        "time_critical": f"minimizes transit time to {int(w_time)} days through hub {w_via}, justifying the freight expenditure",
        "risk_averse": f"prioritizes operational reliability with low combined risk score ({w_risk:.2f}) and minimal delay probability ({w_delay:.0%})",
    }

    focus_desc = profile_descriptions.get(
        profile, "optimizes the designated multi-criteria utility function"
    )

    tie_note = f" [Tied with Route {', '.join(other_tied)}]" if is_tied else ""
    headline = (
        f"Route {w_id} (via {w_via}) recommended for {winner['origin']} → {winner['destination']} "
        f"under '{profile}' profile [{algo}] (Score: {w_score:.1f}/100){tie_note}."
    )

    narrative = (
        f"Under the {profile.replace('_', ' ').upper()} strategy ({algo}), Route {w_id} ranks #1. "
        f"It transits through {w_via} in {int(w_time)} days at a freight cost of ${int(w_cost):,d}, "
        f"with a delay probability of {w_delay:.0%} and combined risk index of {w_risk:.2f}. "
        f"This route {focus_desc}."
    )

    return {
        "shipment_id": str(winner["shipment_id"]),
        "recommended_route": w_id,
        "score": w_score,
        "profile": profile,
        "headline": headline,
        "narrative": narrative,
        "key_tradeoffs": tradeoffs,
        "comparisons": comparisons,
        "is_tied": is_tied,
        "tied_routes": tied_candidates,
    }
