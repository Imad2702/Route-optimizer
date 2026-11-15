# Route Optimizer
### Optimizing Shipment Routes Under Cost, Time, and Risk Constraints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 33 Passed](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![MCDM: WSM & TOPSIS](https://img.shields.io/badge/MCDM-WSM%20%7C%20TOPSIS-orange.svg)]()

---

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Quickstart & Installation](#quickstart--installation)
- [Running the CLI Interface](#running-the-cli-interface)
- [Python API Quickstart](#python-api-quickstart)
- [Running the Streamlit Web Dashboard](#running-the-streamlit-web-dashboard)
- [Optimization Methodology Summary](#optimization-methodology-summary)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview

In global logistics, route selection is fundamentally a decision-making problem under uncertainty. A cheaper route often entails higher delays, while a faster corridor may pass through congested maritime chokepoints or geopolitically sensitive hubs.

This repository implements a **production-grade Multi-Criteria Decision Making (MCDM)** system that:
1. **Models compound multi-leg shipping corridors** (Mainline Ocean Trunk + Regional Feeder approach).
2. **Evaluates routes with dual decision algorithms**:
   - **Weighted Sum Model (WSM)** with Inverted Min-Max relative utility scaling.
   - **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) for geometric compromise ranking.
3. **Recommends optimal routes** tailored to operational priorities (`balanced`, `cost_sensitive`, `time_critical`, `risk_averse`, or custom sliders).
4. **Explains trade-offs** in operator-friendly natural language (quantifying price premiums vs. days saved and risk avoided).
5. **Performs sensitivity analysis** to evaluate recommendation robustness across parameter shifts.
6. **Provides strongly-typed domain entities** and a unified engine orchestrator.
7. **Delivers both an interactive CLI and a rich Streamlit web dashboard**.

---

## Project Architecture

```
Route-optimizer/
├── data/
│   ├── generate_data.py         # Deterministic synthetic dataset generator (seed=42)
│   ├── shipments.csv            # 30 shipments × 4 routes = 120 observations
│   └── shipment_legs.csv        # 240 operational multi-leg records (2 legs per route)
├── optimizer/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Strongly-typed domain models (Leg, Route, Shipment, OptimizationResult)
│   ├── engine.py                # High-level orchestrator & facade (RouteOptimizerEngine)
│   ├── scorer.py                # MCDM scoring engine: Inverted Utility WSM & TOPSIS
│   ├── explainer.py             # Natural language decision justification engine
│   └── sensitivity.py           # Cross-profile sensitivity & robustness analytics
├── tests/
│   ├── test_data_generator.py   # Dataset schema & dimensional validation
│   ├── test_engine.py           # Orchestrator & high-level API tests
│   ├── test_explainer.py        # Natural language narrative & pairwise asserts
│   ├── test_models.py           # Domain entity serialization & risk tests
│   ├── test_scorer.py           # Utility normalization & boundary tests
│   ├── test_sensitivity.py      # Cross-profile stability matrix tests
│   └── test_topsis.py           # TOPSIS algorithm validation tests
├── cli.py                       # Rich terminal command-line interface
├── app.py                       # Interactive Streamlit Web Application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation & instructions
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment recommended

### 2. Clone or Navigate to Directory
```bash
cd Route-optimizer
```

### 3. Create Virtual Environment & Install Dependencies

**Using `uv` (Recommended, ultra-fast):**
```bash
uv venv
uv pip install -r requirements.txt
```

**Or standard `pip`:**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Generate Dataset (Optional - already pre-generated)
```bash
python data/generate_data.py
```
*(Generates 120 route alternatives and 240 operational legs across 30 shipments with realistic trade-off distributions).*

### 5. Run Unit Tests (pytest)
```bash
pytest -v
```
*(Executes 33 unit and integration tests covering scoring, TOPSIS, multi-leg models, orchestrator engine, sensitivity, explanations, tie-breaking, and strict schema validation).*

### 6. Code Quality & Linting (Ruff)
```bash
ruff check .
```
*(Validates codebase with zero linting warnings and full PEP 8 compliance).*

---

## Running the CLI Interface

The CLI uses the `rich` library to deliver formatted terminal tables, multi-leg breakdowns, and decision cards.

### A. Single Shipment Recommendation (Default WSM + Multi-Leg)
```bash
python cli.py --shipment S001 --profile balanced --legs
```
*Options for `--profile`: `balanced`, `cost_sensitive`, `time_critical`, `risk_averse`*

### B. Evaluate with TOPSIS Algorithm
```bash
python cli.py --shipment S001 --algorithm topsis --legs
```

### C. Side-by-Side Algorithm Comparison (WSM vs. TOPSIS)
```bash
python cli.py --shipment S001 --compare-algorithms
```

### D. Compare All Profiles for a Shipment
```bash
python cli.py --shipment S001 --all-profiles
```

### E. Fleet-Wide Recommendation (All 30 Shipments)
```bash
python cli.py --all-shipments --profile time_critical
```

### F. Decision Robustness & Sensitivity Matrix
```bash
python cli.py --sensitivity
```

---

## Python API Quickstart

The optimization engine can also be integrated directly as a Python library:

```python
import pandas as pd
from optimizer.engine import RouteOptimizerEngine

# Initialize the high-level facade
engine = RouteOptimizerEngine()
df = pd.read_csv("data/shipments.csv")

# 1. Optimize a single shipment using TOPSIS and the Balanced profile
result = engine.optimize_shipment(
    df,
    shipment_id="S001",
    profile="balanced",
    algorithm="topsis"
)

winner = result.recommended_route
print(f"Optimal Choice: Route {winner.route_id} via {winner.via}")
print(f"Composite Score: {winner.composite_score:.1f}/100 | Cost: ${winner.cost_usd:,.0f} | Transit: {winner.transit_days}d")
print(f"Decision Narrative: {result.explanation['headline']}")

# 2. Optimize with custom operational weights (must sum to > 0)
custom_weights = {"cost": 0.50, "time": 0.30, "risk": 0.20}
custom_res = engine.optimize_shipment(
    df,
    shipment_id="S001",
    custom_weights=custom_weights,
    algorithm="wsm"
)
print(f"Custom Optimal Choice: Route {custom_res.recommended_route.route_id}")
```

---

## Running the Streamlit Web Dashboard

Launch the interactive dashboard:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.

### Dashboard Features:
1. **Single Shipment Recommendation Tab:**
   - Real-time KPI metric cards (recommended route, score, cost, transit, delay risk).
   - Algorithm switcher: **Weighted Sum Model (WSM)** or **TOPSIS**.
   - **Multi-Leg Operational Breakdown expander** (Mainline Ocean Trunk + Regional Feeder Approach).
   - Candidate route comparison table with winner highlighted in green.
   - Plotly bar chart comparing composite utility scores.
   - Natural language trade-off explanation comparing the winner against the cheapest and runner-up routes.
   - Cross-profile comparison table for the active shipment.
2. **Fleet-Wide Overview Tab:**
   - Fleet aggregate KPIs (mean freight cost, transit days, risk index).
   - Complete table of recommended routes across all 30 shipments.
   - Interactive Plotly donut chart showing route archetype allocation.
3. **Sensitivity & Robustness Analysis Tab:**
   - Interactive matrix mapping route selections across all 4 operational profiles.
   - Decision stability classification (*Robust*, *Moderate*, *Highly Sensitive*).
   - Grouped bar chart comparing route win frequency by profile.
4. **Optimization Methodology Tab:**
   - Mathematical documentation of inverted min-max utility scaling, TOPSIS formulation, and multi-leg compound delay modeling.

---

## Optimization Methodology Summary

### Algorithm 1: Weighted Sum Model (WSM)
Raw metrics are mapped to an inverted dimensionless utility scale $\mu \in [0, 1]$ where 1.0 is the best in that shipment group:
$$\mu_{\text{metric}}(r) = \frac{\max(\text{metric}) - \text{metric}_r}{\max(\text{metric}) - \min(\text{metric})}$$

$$\text{Score}_{\text{WSM}}(r) = 100 \times \left[ w_{\text{cost}} \cdot \mu_{\text{cost}}(r) + w_{\text{time}} \cdot \mu_{\text{time}}(r) + w_{\text{risk}} \cdot \mu_{\text{risk}}(r) \right]$$

### Algorithm 2: TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
1. Vector normalization: $r_{ij} = \frac{x_{ij}}{\sqrt{\sum_k x_{kj}^2}}$
2. Weighted matrix: $v_{ij} = w_j \cdot r_{ij}$
3. Positive and negative ideal solutions: $A_j^+ = \min_i v_{ij}$, $A_j^- = \max_i v_{ij}$
4. Separation measures: $S_i^+ = \sqrt{\sum_j (v_{ij} - A_j^+)^2}, \quad S_i^- = \sqrt{\sum_j (v_{ij} - A_j^-)^2}$
5. Relative closeness to ideal: $\text{Score}_{\text{TOPSIS}}(r) = 100 \times \frac{S_i^-}{S_i^+ + S_i^-}$

### Operational Strategy Profiles
- **Balanced ($w = [0.33, 0.33, 0.34]$)**: General containerized dry freight.
- **Cost-Sensitive ($w = [0.60, 0.20, 0.20]$)**: Price-elastic, non-perishable bulk commodities.
- **Time-Critical ($w = [0.20, 0.60, 0.20]$)**: High-value electronics, seasonal apparel, urgent stock.
- **Risk-Averse ($w = [0.20, 0.20, 0.60]$)**: Pharmaceuticals, cold-chain, high-liability shipments.

---

## Troubleshooting & FAQ

**Q: `ModuleNotFoundError` when running CLI or tests?**  
Ensure your virtual environment is activated, or prefix with your environment's python:  
```bash
.venv\Scripts\python -m pytest -v
```

**Q: Streamlit port conflict or cannot open browser?**  
Specify a custom port:  
```bash
streamlit run app.py --server.port 8502
```

**Q: Do TOPSIS and WSM always recommend the same route?**  
Not necessarily. While they agree in clear-cut dominant scenarios, TOPSIS evaluates relative Euclidean distance to the anti-ideal and ideal vectors, penalizing routes with extreme single-attribute vulnerabilities more heavily than linear WSM. Use `python cli.py --shipment <ID> --compare-algorithms` to inspect differences.




