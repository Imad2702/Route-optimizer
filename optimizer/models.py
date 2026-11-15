"""
Domain Entities and Data Contracts for Logistics Route Optimization.
Provides strongly-typed representations for Legs, Routes, Shipments, and Optimization Results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Leg:
    """
    Represents an individual leg within a multi-leg transportation route.
    """

    leg_index: int
    origin: str
    destination: str
    transport_mode: str  # e.g., 'Ocean Trunk', 'Feeder Vessel', 'Intermodal Rail'
    cost_usd: float
    transit_days: float
    delay_probability: float
    congestion_risk: float
    geopolitical_risk: float

    @property
    def risk_score(self) -> float:
        return round(0.5 * self.congestion_risk + 0.5 * self.geopolitical_risk, 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["risk_score"] = self.risk_score
        return d


@dataclass
class Route:
    """
    Represents an end-to-end candidate route composed of one or more legs.
    """

    route_id: str
    origin: str
    destination: str
    via: str
    cost_usd: float
    transit_days: float
    delay_probability: float
    congestion_risk: float
    geopolitical_risk: float
    risk_score: float
    legs: list[Leg] = field(default_factory=list)

    @property
    def num_legs(self) -> int:
        return len(self.legs) if self.legs else (2 if self.via else 1)

    @classmethod
    def from_row(cls, row: dict[str, Any] | pd.Series, legs: list[Leg] | None = None) -> Route:
        """Create a Route instance from a dataset row or dictionary."""
        row_dict = row.to_dict() if isinstance(row, pd.Series) else row
        route_legs = legs or []
        if not route_legs and row_dict.get("via"):
            # Decompose into 2 synthetic legs if explicit legs not provided
            origin = row_dict["origin"]
            dest = row_dict["destination"]
            via = row_dict["via"]
            c = float(row_dict["cost_usd"])
            t = float(row_dict["transit_days"])
            dp = float(row_dict["delay_probability"])
            cr = float(row_dict["congestion_risk"])
            gr = float(row_dict["geopolitical_risk"])

            leg1 = Leg(
                leg_index=1,
                origin=origin,
                destination=via,
                transport_mode="Ocean Trunk",
                cost_usd=round(c * 0.60, 2),
                transit_days=round(t * 0.55, 1),
                delay_probability=round(dp * 0.60, 2),
                congestion_risk=round(cr * 0.90, 2),
                geopolitical_risk=round(gr * 0.80, 2),
            )
            leg2 = Leg(
                leg_index=2,
                origin=via,
                destination=dest,
                transport_mode="Feeder / Regional",
                cost_usd=round(c * 0.40, 2),
                transit_days=round(t * 0.45, 1),
                delay_probability=round(dp * 0.40, 2),
                congestion_risk=round(cr * 1.10, 2),
                geopolitical_risk=round(gr * 1.20, 2),
            )
            route_legs = [leg1, leg2]

        return cls(
            route_id=str(row_dict["route_id"]),
            origin=str(row_dict["origin"]),
            destination=str(row_dict["destination"]),
            via=str(row_dict.get("via", "")),
            cost_usd=float(row_dict["cost_usd"]),
            transit_days=float(row_dict["transit_days"]),
            delay_probability=float(row_dict["delay_probability"]),
            congestion_risk=float(row_dict["congestion_risk"]),
            geopolitical_risk=float(row_dict["geopolitical_risk"]),
            risk_score=float(row_dict["risk_score"]),
            legs=route_legs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["num_legs"] = self.num_legs
        return data


@dataclass
class ScoredRoute(Route):
    """
    A candidate route evaluated and ranked by an optimization algorithm.
    """

    util_cost: float = 0.0
    util_time: float = 0.0
    util_risk: float = 0.0
    composite_score: float = 0.0
    rank: int = 1
    recommended: bool = False
    evaluated_profile: str = "balanced"
    algorithm: str = "wsm"

    @classmethod
    def from_scored_row(cls, row: dict[str, Any] | pd.Series, legs: list[Leg] | None = None) -> ScoredRoute:
        row_dict = row.to_dict() if isinstance(row, pd.Series) else row
        base_route = Route.from_row(row_dict, legs=legs)
        return cls(
            route_id=base_route.route_id,
            origin=base_route.origin,
            destination=base_route.destination,
            via=base_route.via,
            cost_usd=base_route.cost_usd,
            transit_days=base_route.transit_days,
            delay_probability=base_route.delay_probability,
            congestion_risk=base_route.congestion_risk,
            geopolitical_risk=base_route.geopolitical_risk,
            risk_score=base_route.risk_score,
            legs=base_route.legs,
            util_cost=float(row_dict.get("util_cost", 0.0)),
            util_time=float(row_dict.get("util_time", 0.0)),
            util_risk=float(row_dict.get("util_risk", 0.0)),
            composite_score=float(row_dict.get("composite_score", 0.0)),
            rank=int(row_dict.get("rank", 1)),
            recommended=bool(row_dict.get("recommended", False)),
            evaluated_profile=str(row_dict.get("evaluated_profile", "balanced")),
            algorithm=str(row_dict.get("algorithm", "wsm")),
        )


@dataclass
class Shipment:
    """
    Represents a commercial cargo shipment with origin, destination, and candidate routes.
    """

    shipment_id: str
    origin: str
    destination: str
    routes: list[Route] = field(default_factory=list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, shipment_id: str) -> Shipment:
        subset = df[df["shipment_id"] == shipment_id]
        if subset.empty:
            raise ValueError(f"No shipment found with id '{shipment_id}'")
        first = subset.iloc[0]
        routes = [Route.from_row(r) for _, r in subset.iterrows()]
        return cls(
            shipment_id=shipment_id,
            origin=str(first["origin"]),
            destination=str(first["destination"]),
            routes=routes,
        )

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for r in self.routes:
            rec = r.to_dict()
            rec.pop("legs", None)
            rec["shipment_id"] = self.shipment_id
            records.append(rec)
        return pd.DataFrame(records)


@dataclass
class OptimizationResult:
    """
    Structured outcome of route optimization for a single shipment.
    """

    shipment_id: str
    origin: str
    destination: str
    profile: str
    algorithm: str
    weights: dict[str, float]
    recommended_route: ScoredRoute
    runner_up_route: ScoredRoute | None
    cheapest_route: ScoredRoute
    fastest_route: ScoredRoute
    safest_route: ScoredRoute
    all_routes: list[ScoredRoute]
    scored_df: pd.DataFrame
    explanation: dict[str, Any]
    is_tied: bool = False
    tied_routes: list[str] = field(default_factory=list)

    @property
    def score_gap_to_runner_up(self) -> float:
        if self.runner_up_route:
            return round(self.recommended_route.composite_score - self.runner_up_route.composite_score, 2)
        return 0.0

    @property
    def cost_premium_vs_cheapest(self) -> float:
        return round(self.recommended_route.cost_usd - self.cheapest_route.cost_usd, 2)

    def summary(self) -> str:
        return (
            f"OptimizationResult({self.shipment_id}: {self.origin} -> {self.destination} | "
            f"Winner=Route {self.recommended_route.route_id} via {self.recommended_route.via} | "
            f"Score={self.recommended_route.composite_score:.1f} | Profile={self.profile} | Algo={self.algorithm})"
        )
