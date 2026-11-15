"""
Command-Line Interface (CLI) for Aiqon Route Optimizer.
Provides rich, interactive terminal reporting for shipment route recommendations.
Supports Weighted Sum Model (WSM), TOPSIS, multi-leg breakdowns, and cross-profile sensitivity.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer.engine import RouteOptimizerEngine
from optimizer.scorer import PROFILES, SUPPORTED_ALGORITHMS, get_profile_weights

console = Console()

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "shipments.csv"
)
LEGS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "shipment_legs.csv"
)

engine = RouteOptimizerEngine()


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        console.print(f"[bold red]Dataset not found at {DATA_PATH}![/bold red]")
        console.print("[yellow]Generating dataset automatically...[/yellow]")
        from data.generate_data import main as gen_data

        gen_data()
    return pd.read_csv(DATA_PATH)


def load_legs_dataset() -> pd.DataFrame | None:
    if os.path.exists(LEGS_PATH):
        return pd.read_csv(LEGS_PATH)
    return None


def display_single_shipment(
    df: pd.DataFrame,
    shipment_id: str,
    profile: str = "balanced",
    algorithm: str = "wsm",
    show_legs: bool = False,
):
    try:
        opt_res = engine.optimize_shipment(
            df,
            shipment_id=shipment_id,
            profile=profile,
            algorithm=algorithm,
        )
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

    weights = opt_res.weights
    winner = opt_res.recommended_route
    scored = opt_res.scored_df
    explanation = opt_res.explanation

    # Header Panel
    weight_desc = f"Cost: {weights['cost']:.0%} | Time: {weights['time']:.0%} | Risk: {weights['risk']:.0%}"
    algo_label = "TOPSIS (Closeness to Ideal)" if algorithm.lower() == "topsis" else "WSM (Weighted Sum Utility)"
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Shipment ID:[/bold cyan] [bold white]{shipment_id}[/bold white]  |  "
            f"[bold cyan]Corridor:[/bold cyan] [bold yellow]{opt_res.origin} ➔ {opt_res.destination}[/bold yellow]\n"
            f"[bold cyan]Priority Profile:[/bold cyan] [bold magenta]{profile.upper()}[/bold magenta] ({weight_desc})  |  "
            f"[bold cyan]Algorithm:[/bold cyan] [bold green]{algo_label}[/bold green]",
            title="[bold blue]Aiqon Route Optimization Engine[/bold blue]",
        )
    )

    term_width = console.width or 80
    is_compact = term_width < 85

    # Adaptive Routes Table
    table = Table(
        title=f"Evaluated Route Alternatives ({shipment_id}) [{algorithm.upper()}]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )

    if is_compact:
        table.add_column("Route", justify="left", no_wrap=True)
        table.add_column("Via Hub", justify="left")
        table.add_column("Cost", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("Risk", justify="right", no_wrap=True)
        table.add_column("Score", justify="right", no_wrap=True)

        sorted_routes = scored.sort_values(by="rank")
        for _, row in sorted_routes.iterrows():
            is_rec = row["recommended"]
            route_label = (
                f"[bold green]★ Route {row['route_id']}[/bold green]"
                if is_rec
                else f"  Route {row['route_id']}"
            )
            score_label = (
                f"[bold green]{row['composite_score']:.1f}[/bold green]"
                if is_rec
                else f"{row['composite_score']:.1f}"
            )
            table.add_row(
                route_label,
                str(row["via"]),
                f"${int(row['cost_usd']):,d}",
                f"{int(row['transit_days'])}d",
                f"{row['risk_score']:.2f}",
                score_label,
            )
    else:
        table.add_column("Rank", justify="center", style="dim", no_wrap=True)
        table.add_column("Route", justify="center", no_wrap=True)
        table.add_column("Via Hub", justify="left", no_wrap=True)
        table.add_column("Cost", justify="right", no_wrap=True)
        table.add_column("Transit", justify="right", no_wrap=True)
        table.add_column("Delay", justify="right", no_wrap=True)
        table.add_column("Risk", justify="right", no_wrap=True)
        table.add_column("Score", justify="right", no_wrap=True)
        table.add_column("Status", justify="center", no_wrap=True)

        sorted_routes = scored.sort_values(by="rank")
        for _, row in sorted_routes.iterrows():
            is_rec = row["recommended"]
            style = "bold green" if is_rec else "white"
            status_text = (
                "[bold green]★ RECOMMENDED[/bold green]" if is_rec else "Alternative"
            )

            table.add_row(
                f"#{row['rank']}",
                f"Route {row['route_id']}",
                str(row["via"]),
                f"${int(row['cost_usd']):,d}",
                f"{int(row['transit_days'])}d",
                f"{row['delay_probability']:.0%}",
                f"{row['risk_score']:.2f}",
                f"{row['composite_score']:.1f}",
                status_text,
                style=style,
            )

    console.print(table)

    # Multi-Leg Breakdown Table (if requested or by default for winner)
    if show_legs:
        legs_df = load_legs_dataset()
        if legs_df is not None:
            ship_legs = legs_df[
                (legs_df["shipment_id"] == shipment_id)
                & (legs_df["route_id"] == winner.route_id)
            ]
            if not ship_legs.empty:
                leg_table = Table(
                    title=f"Multi-Leg Breakdown for Recommended Route {winner.route_id} (via {winner.via})",
                    box=box.SIMPLE_HEAD,
                    header_style="bold yellow",
                )
                leg_table.add_column("Leg", justify="center")
                leg_table.add_column("From", justify="left")
                leg_table.add_column("To", justify="left")
                leg_table.add_column("Transport Mode", justify="left")
                leg_table.add_column("Cost", justify="right")
                leg_table.add_column("Transit", justify="right")
                leg_table.add_column("Delay Prob", justify="right")
                leg_table.add_column("Risk Score", justify="right")

                for _, lrow in ship_legs.iterrows():
                    leg_table.add_row(
                        f"Leg {lrow['leg_index']}",
                        str(lrow["from_node"]),
                        str(lrow["to_node"]),
                        str(lrow["transport_mode"]),
                        f"${lrow['cost_usd']:,.0f}",
                        f"{lrow['transit_days']:.1f}d",
                        f"{lrow['delay_probability']:.0%}",
                        f"{lrow['risk_score']:.2f}",
                    )
                console.print(leg_table)

    # Explanation Box
    explanation_body = (
        f"[bold green]▶ Recommendation:[/bold green] {explanation['headline']}\n\n"
        f"[bold]Operator Narrative:[/bold]\n{explanation['narrative']}\n\n"
        f"[bold]Key Trade-Off Factors:[/bold]\n"
    )
    for tradeoff in explanation["key_tradeoffs"]:
        explanation_body += f" • {tradeoff}\n"

    console.print(
        Panel(
            explanation_body.strip(),
            title="[bold green]Decision Intelligence & Trade-Off Analysis[/bold green]",
            border_style="green",
        )
    )
    console.print()


def display_all_profiles_single_shipment(
    df: pd.DataFrame, shipment_id: str, algorithm: str = "wsm"
):
    subset = df[df["shipment_id"] == shipment_id]
    if subset.empty:
        console.print(f"[bold red]Shipment {shipment_id} not found![/bold red]")
        sys.exit(1)

    origin = subset.iloc[0]["origin"]
    destination = subset.iloc[0]["destination"]

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Shipment ID:[/bold cyan] [bold white]{shipment_id}[/bold white]  |  "
            f"[bold cyan]Corridor:[/bold cyan] [bold yellow]{origin} ➔ {destination}[/bold yellow]  |  "
            f"[bold cyan]Algorithm:[/bold cyan] [bold green]{algorithm.upper()}[/bold green]\n"
            f"[dim]Comparing recommendation across all 4 operational priority profiles[/dim]",
            title="[bold blue]Multi-Profile Sensitivity Analysis[/bold blue]",
            expand=False,
        )
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Strategic Profile", justify="left", no_wrap=True)
    table.add_column("Weights (C/T/R)", justify="center", no_wrap=True)
    table.add_column("Top Route", justify="center", no_wrap=True)
    table.add_column("Via Hub", justify="left", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)
    table.add_column("Transit", justify="right", no_wrap=True)
    table.add_column("Risk", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)

    for prof in PROFILES:
        weights = get_profile_weights(prof)
        w_str = f"{weights['cost']:.0%}/{weights['time']:.0%}/{weights['risk']:.0%}"
        opt = engine.optimize_shipment(
            df, shipment_id=shipment_id, profile=prof, algorithm=algorithm
        )
        best = opt.recommended_route

        table.add_row(
            prof.replace("_", " ").title(),
            w_str,
            f"[bold green]Route {best.route_id}[/bold green]",
            str(best.via),
            f"${int(best.cost_usd):,d}",
            f"{int(best.transit_days)}d",
            f"{best.risk_score:.2f}",
            f"{best.composite_score:.1f}",
        )

    console.print(table)
    console.print()


def display_algorithm_comparison(df: pd.DataFrame, shipment_id: str, profile: str = "balanced"):
    """Compare WSM vs TOPSIS side-by-side for a shipment."""
    try:
        comp = engine.compare_algorithms(df, shipment_id=shipment_id, profile=profile)
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

    wsm_res = comp["wsm"]
    top_res = comp["topsis"]

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Shipment ID:[/bold cyan] [bold white]{shipment_id}[/bold white]  |  "
            f"[bold cyan]Corridor:[/bold cyan] [bold yellow]{wsm_res.origin} ➔ {wsm_res.destination}[/bold yellow]  |  "
            f"[bold cyan]Profile:[/bold cyan] [bold magenta]{profile.upper()}[/bold magenta]\n"
            f"[dim]Side-by-side comparison: Weighted Sum Model (WSM) vs. TOPSIS Closeness[/dim]",
            title="[bold blue]MCDM Algorithm Benchmark[/bold blue]",
        )
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Route", justify="center")
    table.add_column("Via Hub", justify="left")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Transit (d)", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("WSM Score", justify="right")
    table.add_column("WSM Rank", justify="center")
    table.add_column("TOPSIS Score", justify="right")
    table.add_column("TOPSIS Rank", justify="center")

    wsm_map = {r.route_id: r for r in wsm_res.all_routes}
    top_map = {r.route_id: r for r in top_res.all_routes}

    for rid in sorted(wsm_map.keys()):
        wr = wsm_map[rid]
        tr = top_map[rid]
        w_star = "★ " if wr.recommended else ""
        t_star = "★ " if tr.recommended else ""

        wsm_rank_str = f"[bold green]{w_star}#{wr.rank}[/bold green]" if wr.recommended else f"#{wr.rank}"
        top_rank_str = f"[bold green]{t_star}#{tr.rank}[/bold green]" if tr.recommended else f"#{tr.rank}"

        table.add_row(
            f"Route {rid}",
            wr.via,
            f"${int(wr.cost_usd):,d}",
            f"{int(wr.transit_days)}",
            f"{wr.risk_score:.2f}",
            f"{wr.composite_score:.1f}",
            wsm_rank_str,
            f"{tr.composite_score:.1f}",
            top_rank_str,
        )

    console.print(table)
    console.print(
        f"[bold]Recommendation Verdict:[/bold] WSM selects [bold green]Route {wsm_res.recommended_route.route_id}[/bold green] | "
        f"TOPSIS selects [bold green]Route {top_res.recommended_route.route_id}[/bold green]\n"
    )


def display_fleet_overview(df: pd.DataFrame, profile: str, algorithm: str = "wsm"):
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Evaluated Profile:[/bold cyan] [bold magenta]{profile.upper()}[/bold magenta]  |  "
            f"[bold cyan]Algorithm:[/bold cyan] [bold green]{algorithm.upper()}[/bold green]\n"
            f"[dim]Analyzing complete fleet of 30 commercial shipments[/dim]",
            title="[bold blue]Fleet-Wide Recommendation Summary[/bold blue]",
        )
    )

    scored = engine.optimize_all(df, profile=profile, algorithm=algorithm)
    winners = scored[scored["recommended"]].copy()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Shipment", justify="center", no_wrap=True)
    table.add_column("Corridor", justify="left", no_wrap=True)
    table.add_column("Route", justify="center", no_wrap=True)
    table.add_column("Via Hub", justify="left", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)
    table.add_column("Transit", justify="right", no_wrap=True)
    table.add_column("Delay", justify="right", no_wrap=True)
    table.add_column("Risk", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)

    for _, row in winners.iterrows():
        table.add_row(
            row["shipment_id"],
            f"{row['origin']} ➔ {row['destination']}",
            f"[bold green]Route {row['route_id']}[/bold green]",
            str(row["via"]),
            f"${int(row['cost_usd']):,d}",
            f"{int(row['transit_days'])}d",
            f"{row['delay_probability']:.0%}",
            f"{row['risk_score']:.2f}",
            f"{row['composite_score']:.1f}",
        )

    console.print(table)

    counts = winners["route_id"].value_counts().sort_index()
    dist_text = "  |  ".join(
        [
            f"Route {k}: [bold]{v}[/bold] ({v / len(winners):.0%})"
            for k, v in counts.items()
        ]
    )
    console.print(f"[bold cyan]Fleet Route Distribution:[/bold cyan] {dist_text}")
    console.print(
        f"[bold cyan]Average Metrics:[/bold cyan] "
        f"Cost: [bold]${winners['cost_usd'].mean():,.0f}[/bold] | "
        f"Transit: [bold]{winners['transit_days'].mean():.1f} days[/bold] | "
        f"Risk Score: [bold]{winners['risk_score'].mean():.2f}[/bold]"
    )
    console.print()


def display_sensitivity_matrix(df: pd.DataFrame):
    sens_df, summary_df = engine.run_sensitivity(df)

    console.print()
    console.print(
        Panel(
            "[bold cyan]Cross-Profile Decision Robustness Matrix[/bold cyan]\n"
            "[dim]Tracking recommendation shift across balanced, cost-sensitive, time-critical, and risk-averse strategies[/dim]",
            title="[bold blue]Sensitivity Analysis Engine[/bold blue]",
        )
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Shipment", justify="center", no_wrap=True)
    table.add_column("Corridor", justify="left", no_wrap=True)
    table.add_column("Balanced", justify="center", no_wrap=True)
    table.add_column("Cost Sens.", justify="center", no_wrap=True)
    table.add_column("Time Crit.", justify="center", no_wrap=True)
    table.add_column("Risk Av.", justify="center", no_wrap=True)
    table.add_column("Robustness", justify="left", no_wrap=True)

    for _, row in sens_df.iterrows():
        st_color = (
            "green"
            if "Robust" in row["stability"]
            else ("yellow" if "Moderate" in row["stability"] else "red")
        )
        table.add_row(
            row["shipment_id"],
            f"{row['origin']} ➔ {row['destination']}",
            f"Route {row['balanced']}",
            f"Route {row['cost_sensitive']}",
            f"Route {row['time_critical']}",
            f"Route {row['risk_averse']}",
            f"[{st_color}]{row['stability']}[/{st_color}]",
        )

    console.print(table)

    console.print("\n[bold cyan]Summary of Route Wins Across Operational Profiles:[/bold cyan]")
    sum_table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    sum_table.add_column("Profile", justify="left", no_wrap=True)
    sum_table.add_column("Route A (Economy)", justify="center", no_wrap=True)
    sum_table.add_column("Route B (Balanced)", justify="center", no_wrap=True)
    sum_table.add_column("Route C (Express)", justify="center", no_wrap=True)
    sum_table.add_column("Route D (Bypass)", justify="center", no_wrap=True)

    for _, row in summary_df.iterrows():
        sum_table.add_row(
            row["profile"].replace("_", " ").title(),
            row["Route_A"],
            row["Route_B"],
            row["Route_C"],
            row["Route_D"],
        )
    console.print(sum_table)
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Aiqon Logistics Route Optimization CLI — Multi-Criteria Decision Framework"
    )
    parser.add_argument(
        "--shipment",
        type=str,
        default=None,
        help="Shipment ID to evaluate (e.g., S001, S005).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["balanced", "cost_sensitive", "time_critical", "risk_averse"],
        help="Operational priority profile.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="wsm",
        choices=list(SUPPORTED_ALGORITHMS),
        help="Optimization algorithm: 'wsm' (Weighted Sum Model) or 'topsis'.",
    )
    parser.add_argument(
        "--legs",
        action="store_true",
        help="Display detailed multi-leg operational breakdown for the recommended route.",
    )
    parser.add_argument(
        "--compare-algorithms",
        action="store_true",
        help="Compare WSM vs. TOPSIS algorithms side-by-side for a shipment.",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Evaluate a single shipment across all priority profiles.",
    )
    parser.add_argument(
        "--all-shipments",
        action="store_true",
        help="Evaluate all 30 shipments under the chosen profile.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run comprehensive sensitivity analysis matrix for all shipments.",
    )

    args = parser.parse_args()
    df = load_dataset()

    # Validate shipment ID upfront if provided
    if args.shipment:
        valid_sids = set(df["shipment_id"].unique())
        if args.shipment not in valid_sids:
            sample_ids = ", ".join(sorted(valid_sids)[:6])
            console.print(
                f"[bold red]Error: Shipment '{args.shipment}' not found in dataset.[/bold red] "
                f"[dim](Available IDs: {sample_ids}...)[/dim]"
            )
            sys.exit(1)

    if args.sensitivity:
        display_sensitivity_matrix(df)
    elif args.all_shipments:
        display_fleet_overview(df, profile=args.profile, algorithm=args.algorithm)
    elif args.compare_algorithms:
        sid = args.shipment or "S001"
        display_algorithm_comparison(df, shipment_id=sid, profile=args.profile)
    elif args.all_profiles:
        sid = args.shipment or "S001"
        display_all_profiles_single_shipment(df, shipment_id=sid, algorithm=args.algorithm)
    elif args.shipment:
        display_single_shipment(
            df,
            shipment_id=args.shipment,
            profile=args.profile,
            algorithm=args.algorithm,
            show_legs=args.legs,
        )
    else:
        # Default behavior: demonstrate S001 balanced recommendation with legs
        display_single_shipment(
            df,
            shipment_id="S001",
            profile=args.profile,
            algorithm=args.algorithm,
            show_legs=True,
        )
        console.print(
            "[dim]Tip: Use `--help` to explore all flags, e.g. `--shipment S002 --compare-algorithms` or `--legs`.[/dim]\n"
        )


if __name__ == "__main__":
    main()
