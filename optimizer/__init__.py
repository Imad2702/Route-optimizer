"""
Logistics Route Optimizer Package.
Multi-Criteria Decision Making (MCDM) under Cost, Time, and Risk Constraints.
"""

from .engine import RouteOptimizerEngine, optimize_shipment
from .explainer import explain_recommendation
from .models import Leg, OptimizationResult, Route, ScoredRoute, Shipment
from .scorer import (
    PROFILES,
    SUPPORTED_ALGORITHMS,
    get_profile_weights,
    recommend_route,
    score_routes,
    validate_shipment_schema,
)
from .sensitivity import (
    profile_distribution_summary,
    run_cross_profile_sensitivity,
    sensitivity_analysis,
)

__all__ = [
    "PROFILES",
    "SUPPORTED_ALGORITHMS",
    "Leg",
    "OptimizationResult",
    "Route",
    "RouteOptimizerEngine",
    "ScoredRoute",
    "Shipment",
    "explain_recommendation",
    "get_profile_weights",
    "optimize_shipment",
    "profile_distribution_summary",
    "recommend_route",
    "run_cross_profile_sensitivity",
    "score_routes",
    "sensitivity_analysis",
    "validate_shipment_schema",
]
