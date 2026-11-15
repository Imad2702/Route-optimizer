"""
Streamlit Web Application: Shipment Route Optimizer
Interactive Decision Intelligence Platform for Global Logistics Route Selection.
Supports Multi-Attribute Utility (WSM), TOPSIS Closeness, and Multi-Leg Operational Analysis.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from optimizer.engine import RouteOptimizerEngine
from optimizer.scorer import PROFILES

st.set_page_config(
    page_title="Aiqon Logistics Route Optimizer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global Custom Styling for Tabs
# ---------------------------------------------------------------------------
st.html(
    """
    <style>
    /* Target the tabs list bar container */
    [data-baseweb="tab-list"], div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        width: 100% !important;
        gap: 16px !important;
        margin-bottom: 16px !important;
    }
    
    /* Target individual tab buttons: spread evenly with card styling */
    [data-baseweb="tab-list"] button, 
    [data-baseweb="tab-list"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"] {
        flex: 1 1 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin: 0 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Hover state */
    [data-baseweb="tab-list"] button:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    
    /* Active / Selected Tab */
    [data-baseweb="tab-list"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.12) !important;
        border-color: #ff4b4b !important;
        color: #ff6b6b !important;
        font-weight: 600 !important;
    }
    
    /* Hide the red underline bar so it looks like clean separate cards */
    [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* Tab label text styling */
    [data-baseweb="tab-list"] button p {
        font-size: 14px !important;
        margin: 0 !important;
    }

    /* Tighten Sidebar layout to prevent scrolling */
    section[data-testid="stSidebar"] div.stSidebarUserContent {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
    }
    section[data-testid="stSidebar"] .stElementContainer {
        margin-bottom: 0.35rem !important;
    }
    section[data-testid="stSidebar"] label {
        margin-bottom: 0.15rem !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    </style>
    """
)

engine = RouteOptimizerEngine()


# ---------------------------------------------------------------------------
# Data Loading & Cached Analytics
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(CURRENT_DIR, "data", "shipments.csv")
    legs_path = os.path.join(CURRENT_DIR, "data", "shipment_legs.csv")
    if not os.path.exists(csv_path) or not os.path.exists(legs_path):
        from data.generate_data import main as gen_data

        gen_data()
    routes_df = pd.read_csv(csv_path)
    legs_df = pd.read_csv(legs_path) if os.path.exists(legs_path) else None
    return routes_df, legs_df


@st.cache_data
def get_cached_sensitivity(routes_df: pd.DataFrame):
    """Caches cross-profile sensitivity analysis to eliminate tab re-computation lag."""
    return engine.run_sensitivity(routes_df)


df, legs_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <img src="https://img.icons8.com/fluency/96/cargo-ship.png" width="38" style="vertical-align: middle;">
        <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; line-height: 1.1;">Route Optimizer</span>
    </div>
    <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 14px;">Multi-Criteria Logistics Decision Intelligence</div>
    """,
    unsafe_allow_html=True,
)

# 1. Shipment Selector (Moved to Top for Fast Dispatching)
unique_shipments = df[["shipment_id", "origin", "destination"]].drop_duplicates()
shipment_labels = {
    row["shipment_id"]: f"{row['shipment_id']} ({row['origin']} → {row['destination']})"
    for _, row in unique_shipments.iterrows()
}
selected_sid = st.sidebar.selectbox(
    "Choose Shipment",
    options=unique_shipments["shipment_id"].tolist(),
    format_func=lambda x: shipment_labels[x],
    index=0,
)

# 2. Optimization Algorithm
selected_algorithm = st.sidebar.radio(
    "Algorithm",
    ["Weighted Sum (WSM)", "TOPSIS"],
    index=0,
    help="WSM: Weighted utility score. TOPSIS: Best balanced compromise.",
)
algo_code = "topsis" if "TOPSIS" in selected_algorithm else "wsm"

# 3. Operational Strategy
mode = st.sidebar.radio(
    "Strategy Mode",
    ["Standard Profiles", "Custom Weights"],
    horizontal=True,
    help="Choose between pre-tuned business profiles or fine-grained weight control.",
)

custom_weights = None
if mode == "Standard Profiles":
    profile_choice = st.sidebar.selectbox(
        "Priority Profile",
        options=list(PROFILES.keys()),
        format_func=lambda x: f"{x.replace('_', ' ').title()}",
        index=0,
    )
    profile_info = PROFILES[profile_choice]
    st.sidebar.caption(
        f"**Weights:** Cost {profile_info['cost']:.0%} • Time {profile_info['time']:.0%} • Risk {profile_info['risk']:.0%}"
    )
    selected_profile = profile_choice
else:
    selected_profile = "custom"
    w_cost = st.sidebar.slider("Cost Weight", 0.0, 1.0, 0.4, 0.05)
    w_time = st.sidebar.slider("Transit Time Weight", 0.0, 1.0, 0.3, 0.05)
    w_risk = st.sidebar.slider("Operational Risk Weight", 0.0, 1.0, 0.3, 0.05)

    total_w = w_cost + w_time + w_risk
    if total_w == 0:
        st.sidebar.error("Warning: All weights are 0!")
        custom_weights = None
        weights_are_zero = True
    else:
        custom_weights = {
            "cost": w_cost / total_w,
            "time": w_time / total_w,
            "risk": w_risk / total_w,
        }
        weights_are_zero = False
        st.sidebar.caption(
            f"Normalized: {custom_weights['cost']:.0%} Cost • {custom_weights['time']:.0%} Time • {custom_weights['risk']:.0%} Risk"
        )

# ---------------------------------------------------------------------------
# Zero-Weight Validation Guard
# ---------------------------------------------------------------------------
if mode == "Custom Criterion Weights" and weights_are_zero:
    st.warning(
        "### Warning: Optimization Paused — All Weights Are Set to Zero\n\n"
        "A multi-criteria route recommendation cannot be computed when **Cost**, **Transit Time**, and **Operational Risk** are all set to **0%**.\n\n"
        "**To continue:**\n"
        "- **Adjust sliders:** Increase at least one slider in the left sidebar above **0.0** (e.g., set Cost, Time, or Risk weight), or\n"
        "- **Use presets:** Switch the **Strategy Mode** in the sidebar back to **Standard Profiles** (Balanced, Cost-Sensitive, Time-Critical, or Risk-Averse)."
    )
    st.info(
        "*In logistics operations, every routing choice requires at least one guiding priority constraint.*"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Single Shipment Recommendation",
        "Fleet-Wide Overview",
        "Sensitivity & Robustness Analysis",
        "Optimization Methodology",
    ]
)

# ===========================================================================
# TAB 1: Single Shipment Recommendation
# ===========================================================================
with tab1:

    opt_result = engine.optimize_shipment(
        df,
        shipment_id=selected_sid,
        profile=selected_profile if custom_weights is None else "balanced",
        custom_weights=custom_weights,
        algorithm=algo_code,
    )
    winner = opt_result.recommended_route
    scored_shipment = opt_result.scored_df
    explanation = opt_result.explanation

    st.markdown(
    f"### Evaluation for **{selected_sid}**: **{winner.origin} → {winner.destination}** <small style='color:#666'>{selected_algorithm}</small>",
    unsafe_allow_html=True,
)

    # Key Metric Cards
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric(
            label="Recommended Route",
            value=f"Route {winner.route_id}",
            delta=f"Via {winner.via}",
            delta_color="normal",
        )
    with col_m2:
        st.metric(
            label="Composite Score", value=f"{winner.composite_score:.1f} / 100"
        )
    with col_m3:
        st.metric(label="Freight Cost", value=f"${int(winner.cost_usd):,d}")
    with col_m4:
        st.metric(label="Transit Duration", value=f"{int(winner.transit_days)} days")
    with col_m5:
        st.metric(
            label="Operational Risk Index",
            value=f"{winner.risk_score:.2f}",
            help=f"Route Delay Probability: {winner.delay_probability:.0%}",
        )

    # Explanation Callout Box
    st.success(
        f"**Recommendation Summary:** {explanation['headline']}\n\n"
        f"**Decision Narrative:** {explanation['narrative']}"
    )

    with st.expander(
        "View Detailed Trade-Off Breakdown vs. Alternatives", expanded=True
    ):
        for bullet in explanation["key_tradeoffs"]:
            st.markdown(f"- {bullet}")

    # Multi-Leg Operational Breakdown Expander
    if legs_df is not None:
        ship_legs = legs_df[
            (legs_df["shipment_id"] == selected_sid)
            & (legs_df["route_id"] == winner.route_id)
        ]
        if not ship_legs.empty:
            with st.expander(
                f"Multi-Leg Operational Breakdown for Route {winner.route_id} (via {winner.via})",
                expanded=False,
            ):
                st.caption(
                    "Shipment route broken down into discrete maritime legs: Mainline Ocean Trunk & Regional Feeder Approach."
                )
                leg_view = ship_legs[
                    [
                        "leg_index",
                        "from_node",
                        "to_node",
                        "transport_mode",
                        "cost_usd",
                        "transit_days",
                        "delay_probability",
                        "congestion_risk",
                        "geopolitical_risk",
                        "risk_score",
                    ]
                ].copy()
                leg_view.columns = [
                    "Leg #",
                    "From Hub",
                    "To Hub",
                    "Transport Mode",
                    "Cost ($)",
                    "Transit (days)",
                    "Delay Prob",
                    "Congestion",
                    "Geopolitical",
                    "Risk Score",
                ]
                leg_view["Cost ($)"] = leg_view["Cost ($)"].apply(lambda x: f"${x:,.0f}")
                leg_view["Transit (days)"] = leg_view["Transit (days)"].apply(lambda x: f"{x:.1f}d")
                leg_view["Delay Prob"] = leg_view["Delay Prob"].apply(lambda x: f"{x:.0%}")
                st.dataframe(leg_view, width="stretch", hide_index=True)

    # Visual Route Comparison & Chart
    col_table, col_chart = st.columns([1.1, 1.0])

    with col_table:
        active_prof_title = (
            f"Custom ({custom_weights['cost']:.0%}/{custom_weights['time']:.0%}/{custom_weights['risk']:.0%})"
            if custom_weights
            else selected_profile.replace("_", " ").title()
        )
        st.subheader("Candidate Routes Comparison")
        st.caption(f"Currently evaluated under strategy: **{active_prof_title}** • {algo_code.upper()}")
        display_cols = [
            "route_id",
            "via",
            "cost_usd",
            "transit_days",
            "delay_probability",
            "risk_score",
            "composite_score",
            "rank",
        ]
        table_df = scored_shipment[display_cols].sort_values(by="rank").copy()
        table_df.columns = [
            "Route",
            "Via Hub",
            "Cost ($)",
            "Transit (days)",
            "Delay Prob",
            "Risk Index",
            "Score (0-100)",
            "Rank",
        ]
        table_df["Delay Prob"] = table_df["Delay Prob"].apply(lambda x: f"{x:.0%}")
        table_df["Cost ($)"] = table_df["Cost ($)"].apply(lambda x: f"${int(x):,d}")
        table_df["Risk Index"] = table_df["Risk Index"].apply(lambda x: f"{x:.2f}")
        table_df["Score (0-100)"] = table_df["Score (0-100)"].apply(lambda x: f"{x:.1f}")

        def highlight_winner(row):
            if row["Rank"] == 1:
                return [
                    "background-color: #d1fae5; font-weight: bold; color: #065f46"
                ] * len(row)
            return [""] * len(row)

        st.dataframe(
            table_df.style.apply(highlight_winner, axis=1),
            width="stretch",
            hide_index=True,
        )

    with col_chart:
        st.subheader("Composite Score Comparison")
        chart_df = scored_shipment.sort_values(by="route_id").copy()
        chart_df["Route Label"] = chart_df.apply(
            lambda r: f"Route {r['route_id']} ({r['via']})", axis=1
        )
        colors = ["#10b981" if r else "#94a3b8" for r in chart_df["recommended"]]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=chart_df["Route Label"],
                    y=chart_df["composite_score"],
                    marker_color=colors,
                    text=chart_df["composite_score"].apply(lambda v: f"{v:.1f}"),
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title=f"Route Utility Scores ({algo_code.upper()})",
            yaxis={"title": "Score (0-100)", "range": [0, 105]},
            xaxis={"title": "Route Candidate"},
            height=320,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        st.plotly_chart(fig, width="stretch")

    # Multi-Profile comparison for this shipment

    st.subheader(
        f"Strategic Benchmark: How recommendations for {selected_sid} shift across profiles"
    )
    st.caption(
        "**Note:** This table compares all operational strategies simultaneously. "
        "The row matching your currently selected sidebar profile is highlighted below."
    )

    shipment_raw = df[df["shipment_id"] == selected_sid].copy()
    multi_records = []
    for prof in PROFILES:
        opt_b = engine.optimize_shipment(
            shipment_raw, profile=prof, algorithm=algo_code
        )
        best_r = opt_b.recommended_route
        is_active = selected_profile == prof and custom_weights is None
        multi_records.append(
            {
                "Selection": "ACTIVE" if is_active else "",
                "Strategic Profile": prof.replace("_", " ").title(),
                "Recommended Route": f"Route {best_r.route_id}",
                "Via Hub": best_r.via,
                "Freight Cost ($)": f"${int(best_r.cost_usd):,d}",
                "Transit Days": f"{int(best_r.transit_days)} days",
                "Delay Prob": f"{best_r.delay_probability:.0%}",
                "Risk Index": f"{best_r.risk_score:.2f}",
                "Score": f"{best_r.composite_score:.1f}",
                "_is_active": is_active,
            }
        )

    # Include custom profile row if active
    if custom_weights is not None:
        opt_c = engine.optimize_shipment(
            shipment_raw, custom_weights=custom_weights, algorithm=algo_code
        )
        best_c = opt_c.recommended_route
        multi_records.append(
            {
                "Selection": "ACTIVE",
                "Strategic Profile": f"Custom ({custom_weights['cost']:.0%} Cost / {custom_weights['time']:.0%} Time / {custom_weights['risk']:.0%} Risk)",
                "Recommended Route": f"Route {best_c.route_id}",
                "Via Hub": best_c.via,
                "Freight Cost ($)": f"${int(best_c.cost_usd):,d}",
                "Transit Days": f"{int(best_c.transit_days)} days",
                "Delay Prob": f"{best_c.delay_probability:.0%}",
                "Risk Index": f"{best_c.risk_score:.2f}",
                "Score": f"{best_c.composite_score:.1f}",
                "_is_active": True,
            }
        )

    display_benchmark = pd.DataFrame(multi_records).drop(columns=["_is_active"])

    def highlight_active_profile(row):
        if row["Selection"] == "ACTIVE":
            return [
                "background-color: #dbeafe; font-weight: bold; color: #1e40af"
            ] * len(row)
        return [""] * len(row)

    styled_benchmark = display_benchmark.style.apply(highlight_active_profile, axis=1)
    st.dataframe(styled_benchmark, width="stretch", hide_index=True)

# ===========================================================================
# TAB 2: Fleet-Wide Overview
# ===========================================================================
with tab2:

    st.subheader(f"Fleet-Wide Route Optimization: {selected_algorithm} (30 Shipments)")

    fleet_scored = engine.optimize_all(
        df,
        profile=selected_profile if custom_weights is None else "balanced",
        custom_weights=custom_weights,
        algorithm=algo_code,
    )
    fleet_winners = fleet_scored[fleet_scored["recommended"]].copy()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Active Shipments", len(fleet_winners))
    with kpi2:
        st.metric("Average Freight Cost", f"${fleet_winners['cost_usd'].mean():,.0f}")
    with kpi3:
        st.metric(
            "Average Transit Time", f"{fleet_winners['transit_days'].mean():.1f} days"
        )
    with kpi4:
        st.metric("Average Risk Score", f"{fleet_winners['risk_score'].mean():.2f}")

    col_f1, col_f2 = st.columns([1.2, 0.8])
    with col_f1:
        st.markdown("#### Selected Routes Across Fleet")
        fleet_display = fleet_winners[
            [
                "shipment_id",
                "origin",
                "destination",
                "route_id",
                "via",
                "cost_usd",
                "transit_days",
                "risk_score",
                "composite_score",
            ]
        ].copy()
        fleet_display.columns = [
            "Shipment",
            "Origin",
            "Destination",
            "Route",
            "Via Hub",
            "Cost ($)",
            "Transit (d)",
            "Risk Index",
            "Score",
        ]
        fleet_display["Cost ($)"] = fleet_display["Cost ($)"].apply(
            lambda x: f"${int(x):,d}"
        )
        fleet_display["Risk Index"] = fleet_display["Risk Index"].apply(
            lambda x: f"{x:.2f}"
        )
        fleet_display["Score"] = fleet_display["Score"].apply(
            lambda x: f"{x:.1f}"
        )
        st.dataframe(
            fleet_display, width="stretch", height=400, hide_index=True
        )

    with col_f2:
        st.markdown("#### Route Type Allocation Breakdown")
        route_counts = fleet_winners["route_id"].value_counts().reset_index()
        route_counts.columns = ["Route Type", "Count"]

        type_descriptions = {
            "A": "Route A (Economy / High Risk)",
            "B": "Route B (Balanced Compromise)",
            "C": "Route C (Express / Low Risk)",
            "D": "Route D (Mid-Tier / Low Volatility)",
        }
        route_counts["Description"] = route_counts["Route Type"].map(type_descriptions)

        fig_pie = px.pie(
            route_counts,
            values="Count",
            names="Description",
            color="Route Type",
            color_discrete_map={
                "A": "#f59e0b",
                "B": "#3b82f6",
                "C": "#10b981",
                "D": "#8b5cf6",
            },
            hole=0.4,
        )
        fig_pie.update_layout(height=400, margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig_pie, width="stretch")

# ===========================================================================
# TAB 3: Sensitivity & Robustness Analysis
# ===========================================================================
with tab3:

    st.subheader("Decision Robustness Matrix Across 4 Strategic Profiles")
    st.write(
        "Demonstrates how sensitive route choices are when priorities shift from "
        "cost minimization to time urgency or risk aversion."
    )

    sens_df, summary_df = get_cached_sensitivity(df)

    col_s1, col_s2 = st.columns([1.4, 0.6])
    with col_s1:
        st.markdown("#### Multi-Profile Routing Matrix")
        st.dataframe(
            sens_df[
                [
                    "shipment_id",
                    "origin",
                    "destination",
                    "balanced",
                    "cost_sensitive",
                    "time_critical",
                    "risk_averse",
                    "stability",
                ]
            ],
            width="stretch",
            height=380,
            hide_index=True,
        )

    with col_s2:
        st.markdown("#### Decision Stability Distribution")
        stability_counts = sens_df["stability"].value_counts().reset_index()
        stability_counts.columns = ["Stability Level", "Count"]

        fig_stab = px.bar(
            stability_counts,
            x="Stability Level",
            y="Count",
            color="Stability Level",
            color_discrete_map={
                "Very High (Robust)": "#10b981",
                "Moderate": "#f59e0b",
                "Highly Sensitive": "#ef4444",
            },
        )
        fig_stab.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_stab, width="stretch")


    st.subheader("Aggregate Win Rate by Operational Profile")

    chart_data = []
    for prof in PROFILES:
        for r in ["A", "B", "C", "D"]:
            c = sens_df[sens_df[prof] == r].shape[0]
            chart_data.append(
                {
                    "Profile": prof.replace("_", " ").title(),
                    "Route": f"Route {r}",
                    "Shipments Won": c,
                }
            )
    chart_cdf = pd.DataFrame(chart_data)

    fig_grouped = px.bar(
        chart_cdf,
        x="Profile",
        y="Shipments Won",
        color="Route",
        barmode="group",
        color_discrete_map={
            "Route A": "#f59e0b",
            "Route B": "#3b82f6",
            "Route C": "#10b981",
            "Route D": "#8b5cf6",
        },
    )
    fig_grouped.update_layout(
        height=380, yaxis={"title": "Number of Shipments (out of 30)"}
    )
    st.plotly_chart(fig_grouped, width="stretch")

# ===========================================================================
# TAB 4: Optimization Methodology
# ===========================================================================
with tab4:

    st.subheader("Multi-Criteria Decision Making (MCDM) Framework")
    st.markdown("""
    ### 1. Problem Formulation
    Shipment route selection is formulated as a multi-attribute utility optimization problem under competing objectives:
    - **Minimize Freight Cost ($C$)**: Direct logistics carrier and bunker expenses.
    - **Minimize Transit Duration ($T$)**: Port-to-port and inland transit timeline.
    - **Minimize Operational Risk ($R$)**: Composite indicator of port congestion and geopolitical vulnerability.

    ### 2. Available Optimization Algorithms

    #### A. Weighted Sum Model (WSM) — Inverted Relative Utility
    Candidate routes within each shipment $S_i$ are mapped to a dimensionless utility scale $\\mu \\in [0, 1]$:
    $$\\mu_{cost}(r) = \\frac{C_{max} - C_r}{C_{max} - C_{min}}, \\quad \\mu_{time}(r) = \\frac{T_{max} - T_r}{T_{max} - T_{min}}, \\quad \\mu_{risk}(r) = \\frac{R_{max} - R_r}{R_{max} - R_{min}}$$

    $$\\text{Score}_{\\text{WSM}}(r) = 100 \\times \\left[ w_{cost} \\cdot \\mu_{cost}(r) + w_{time} \\cdot \\mu_{time}(r) + w_{risk} \\cdot \\mu_{risk}(r) \\right]$$

    #### B. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    Evaluates geometric closeness to the positive-ideal compromise solution:
    1. **Vector Normalization:** $r_{ij} = \\frac{x_{ij}}{\\sqrt{\\sum_k x_{kj}^2}}$
    2. **Weighted Normalized Matrix:** $v_{ij} = w_j \\cdot r_{ij}$
    3. **Ideal Best ($A^+$) and Ideal Worst ($A^-$):** $A_j^+ = \\min_i v_{ij}$, $A_j^- = \\max_i v_{ij}$ (cost criteria)
    4. **Euclidean Distances:** $S_i^+ = \\sqrt{\\sum_j (v_{ij} - A_j^+)^2}, \\quad S_i^- = \\sqrt{\\sum_j (v_{ij} - A_j^-)^2}$
    5. **Relative Closeness Score:** $\\text{Score}_{\\text{TOPSIS}}(r) = 100 \\times \\frac{S_i^-}{S_i^+ + S_i^-}$

    ### 3. Multi-Leg Modeling Representation
    Every end-to-end corridor option is modeled as a compound chain:
    - **Leg 1 (Mainline Trunk):** Origin port to intermediate transshipment hub.
    - **Leg 2 (Regional Feeder):** Hub to destination port.
    Network reliability is maintained via compound delay probability: $P(\\text{delay}) = 1 - (1 - p_1)(1 - p_2)$.

    ### 4. Operational Profiles
    | Profile | Cost Weight ($w_C$) | Time Weight ($w_T$) | Risk Weight ($w_R$) | Operational Context |
    |---|---|---|---|---|
    | **Balanced** | 33% | 33% | 34% | Standard commercial dry goods, flexible schedules |
    | **Cost Sensitive** | 60% | 20% | 20% | Non-perishable bulk commodities, high margin sensitivity |
    | **Time Critical** | 20% | 60% | 20% | High-value electronics, seasonal apparel, urgent stock replenishments |
    | **Risk Averse** | 20% | 20% | 60% | Critical pharmaceuticals, cold-chain, high-liability shipments |

    ### 5. Automated Decision Explainability
    Unlike black-box models, this engine dynamically extracts relative delta metrics:
    - Marginal cost per transit day saved: $\\frac{\\Delta \\text{Cost}}{\\Delta \\text{Days}}$
    - Risk reduction percentage relative to lowest-cost base route
    - Pairwise comparison vs closest runner-up alternative
    """)

st.markdown("---")
st.caption("Route Optimizer")
