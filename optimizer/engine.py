"""
Unified Route Optimization Engine.
Serves as the high-level orchestrator binding scoring, multi-criteria decision algorithms,
explainability, and domain entities into a cohesive, production-grade interface.
"""

from __future__ import annotations

import pandas as pd

from .explainer import explain_recommendation
from .models import OptimizationResult, ScoredRoute, Shipment
from .scorer import (
    get_profile_weights,
    score_routes,
    validate_shipment_schema,
)
from .sensitivity import profile_distribution_summary, run_cross_profile_sensitivity


class RouteOptimizerEngine:
    """
    Unified Orchestrator for evaluating and recommending global freight shipment routes.

    Parameters:
    -----------
    default_algorithm : str
        Default decision algorithm ('wsm' or 'topsis'). Defaults to 'wsm'.
    default_profile : str
        Default operational profile ('balanced', 'cost_sensitive', 'time_critical', 'risk_averse').
    """

    def __init__(
        self,
        default_algorithm: str = "wsm",
        default_profile: str = "balanced",
    ) -> None:
        self.default_algorithm = default_algorithm
        self.default_profile = default_profile

    def optimize_shipment(
        self,
        data: pd.DataFrame | Shipment,
        shipment_id: str | None = None,
        profile: str | None = None,
        custom_weights: dict[str, float] | None = None,
        algorithm: str | None = None,
    ) -> OptimizationResult:
        """
        Evaluate and recommend the optimal route for a single shipment with full explainability.

        Returns:
        --------
        OptimizationResult:
            Richly-typed result object containing winner, runner-up, scored routes, and explanation.
        """
        algo = (algorithm or self.default_algorithm).lower().strip()
        prof = (profile or self.default_profile).lower().strip()

        # Convert Shipment model to DataFrame if needed
        if isinstance(data, Shipment):
            shipment_df = data.to_dataframe()
            target_sid = data.shipment_id
        else:
            validate_shipment_schema(data)
            if shipment_id is not None:
                shipment_df = data[data["shipment_id"] == shipment_id].copy()
                if shipment_df.empty:
                    raise ValueError(f"Shipment '{shipment_id}' not found in dataset.")
                target_sid = shipment_id
            else:
                unique_sids = data["shipment_id"].unique()
                if len(unique_sids) != 1:
                    raise ValueError(
                        f"Expected exactly 1 shipment, but found {len(unique_sids)}. "
                        "Please specify `shipment_id`."
                    )
                target_sid = unique_sids[0]
                shipment_df = data.copy()

        # Score routes with requested MCDM algorithm
        scored_df = score_routes(
            shipment_df,
            profile=prof,
            custom_weights=custom_weights,
            algorithm=algo,
        )

        # Generate operator-friendly natural language explanation
        explanation = explain_recommendation(scored_df, profile=prof)

        # Extract benchmark routes
        sorted_df = scored_df.sort_values(by="rank").reset_index(drop=True)
        all_scored_routes = [
            ScoredRoute.from_scored_row(row) for _, row in sorted_df.iterrows()
        ]

        winner_route = all_scored_routes[0]
        runner_up_route = all_scored_routes[1] if len(all_scored_routes) > 1 else None

        cheapest_row = scored_df.loc[scored_df["cost_usd"].idxmin()]
        fastest_row = scored_df.loc[scored_df["transit_days"].idxmin()]
        safest_row = scored_df.loc[scored_df["risk_score"].idxmin()]

        weights = get_profile_weights(profile=prof, custom_weights=custom_weights)

        return OptimizationResult(
            shipment_id=target_sid,
            origin=str(winner_route.origin),
            destination=str(winner_route.destination),
            profile=prof,
            algorithm=algo,
            weights=weights,
            recommended_route=winner_route,
            runner_up_route=runner_up_route,
            cheapest_route=ScoredRoute.from_scored_row(cheapest_row),
            fastest_route=ScoredRoute.from_scored_row(fastest_row),
            safest_route=ScoredRoute.from_scored_row(safest_row),
            all_routes=all_scored_routes,
            scored_df=scored_df,
            explanation=explanation,
            is_tied=bool(explanation.get("is_tied", False)),
            tied_routes=list(explanation.get("tied_routes", [])),
        )

    def optimize_all(
        self,
        df: pd.DataFrame,
        profile: str | None = None,
        custom_weights: dict[str, float] | None = None,
        algorithm: str | None = None,
    ) -> pd.DataFrame:
        """
        Fleet-wide route optimization across all shipments in dataset.
        """
        algo = (algorithm or self.default_algorithm).lower().strip()
        prof = (profile or self.default_profile).lower().strip()
        return score_routes(
            df,
            profile=prof,
            custom_weights=custom_weights,
            algorithm=algo,
        )

    def run_sensitivity(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs complete cross-profile decision sensitivity and generates distribution summary.
        """
        sens_df = run_cross_profile_sensitivity(df)
        dist_df = profile_distribution_summary(sens_df)
        return sens_df, dist_df

    def compare_algorithms(
        self,
        df: pd.DataFrame,
        shipment_id: str,
        profile: str = "balanced",
    ) -> dict[str, OptimizationResult]:
        """
        Run both WSM and TOPSIS on the same shipment and return side-by-side results.
        """
        res_wsm = self.optimize_shipment(
            df, shipment_id=shipment_id, profile=profile, algorithm="wsm"
        )
        res_topsis = self.optimize_shipment(
            df, shipment_id=shipment_id, profile=profile, algorithm="topsis"
        )
        return {"wsm": res_wsm, "topsis": res_topsis}


# Module-level convenience singleton
_default_engine = RouteOptimizerEngine()


def optimize_shipment(
    data: pd.DataFrame | Shipment,
    shipment_id: str | None = None,
    profile: str = "balanced",
    custom_weights: dict[str, float] | None = None,
    algorithm: str = "wsm",
) -> OptimizationResult:
    """Convenience functional interface for single shipment optimization."""
    return _default_engine.optimize_shipment(
        data=data,
        shipment_id=shipment_id,
        profile=profile,
        custom_weights=custom_weights,
        algorithm=algorithm,
    )
