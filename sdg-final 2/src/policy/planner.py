"""
Intervention graph ("skill-tree") and recommendation planner.
Defines interventions as a DAG with prerequisites, estimates effects,
and generates ranked roadmaps.
"""

import numpy as np
import pandas as pd
import networkx as nx
import json


# ─── Intervention definitions (v0.1) ────────────────────────────────
INTERVENTIONS = [
    {
        "tech_id": "GRID_UPGRADE",
        "name": "Grid T&D Upgrade",
        "family": "infrastructure",
        "maturity_level": "mature",
        "lead_time_years": 3,
        "lifetime_years": 30,
        "adoption_metric": "grid_losses_pct",
        "sdg_links": ["SDG7", "SDG9"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["gov_effectiveness > 0.3"],
        "estimated_effects": {
            "renewables_share_pct": 2.0,
            "electricity_access_pct": 3.0,
            "co2_per_capita_t": -0.1,
            "energy_intensity_mj_usd": -0.3,
        },
        "cost_category": "high",
        "description": "Modernize transmission & distribution to reduce losses and enable variable RE integration."
    },
    {
        "tech_id": "SOLAR_SCALE",
        "name": "Solar PV Scale-up",
        "family": "renewables",
        "maturity_level": "mature",
        "lead_time_years": 1,
        "lifetime_years": 25,
        "adoption_metric": "solar_share_pct",
        "sdg_links": ["SDG7", "SDG13"],
        "prerequisites_hard": ["GRID_UPGRADE"],
        "prerequisites_soft": ["regulatory_quality > 0.2"],
        "estimated_effects": {
            "renewables_share_pct": 5.0,
            "co2_per_capita_t": -0.5,
            "solar_share_pct": 8.0,
        },
        "cost_category": "medium",
        "description": "Utility-scale and distributed solar deployment via auctions and feed-in mechanisms."
    },
    {
        "tech_id": "WIND_SCALE",
        "name": "Wind Power Scale-up",
        "family": "renewables",
        "maturity_level": "mature",
        "lead_time_years": 2,
        "lifetime_years": 25,
        "adoption_metric": "wind_share_pct",
        "sdg_links": ["SDG7", "SDG13"],
        "prerequisites_hard": ["GRID_UPGRADE"],
        "prerequisites_soft": ["regulatory_quality > 0.2"],
        "estimated_effects": {
            "renewables_share_pct": 5.0,
            "co2_per_capita_t": -0.5,
            "wind_share_pct": 8.0,
        },
        "cost_category": "medium",
        "description": "Onshore and offshore wind deployment with grid integration support."
    },
    {
        "tech_id": "COAL_PHASEDOWN",
        "name": "Coal Phase-down",
        "family": "fossil_exit",
        "maturity_level": "policy",
        "lead_time_years": 3,
        "lifetime_years": 20,
        "adoption_metric": "coal_share_pct",
        "sdg_links": ["SDG13", "SDG7"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["gov_effectiveness > 0.4", "capability_index > 0.4"],
        "estimated_effects": {
            "co2_per_capita_t": -1.5,
            "coal_share_pct": -15.0,
            "renewables_share_pct": 3.0,
        },
        "cost_category": "high",
        "description": "Managed retirement of coal plants with just transition support."
    },
    {
        "tech_id": "BUILDING_EFFICIENCY",
        "name": "Building Efficiency (Retrofits)",
        "family": "efficiency",
        "maturity_level": "mature",
        "lead_time_years": 2,
        "lifetime_years": 20,
        "adoption_metric": "energy_intensity_mj_usd",
        "sdg_links": ["SDG7", "SDG13", "SDG9"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["gdp_pc_ppp > 5000"],
        "estimated_effects": {
            "energy_intensity_mj_usd": -0.8,
            "co2_per_capita_t": -0.3,
        },
        "cost_category": "medium",
        "description": "Insulation, windows, HVAC upgrades in existing buildings."
    },
    {
        "tech_id": "HEAT_PUMPS",
        "name": "Heat Pump Electrification",
        "family": "electrification",
        "maturity_level": "scaling",
        "lead_time_years": 2,
        "lifetime_years": 15,
        "adoption_metric": "heat_pump_adoption_proxy",
        "sdg_links": ["SDG7", "SDG13"],
        "prerequisites_hard": ["GRID_UPGRADE"],
        "prerequisites_soft": ["gdp_pc_ppp > 10000"],
        "estimated_effects": {
            "energy_intensity_mj_usd": -0.5,
            "co2_per_capita_t": -0.4,
        },
        "cost_category": "medium",
        "description": "Replace fossil heating with air/ground-source heat pumps."
    },
    {
        "tech_id": "CARBON_PRICING",
        "name": "Carbon Pricing (ETS/Tax)",
        "family": "policy",
        "maturity_level": "policy",
        "lead_time_years": 2,
        "lifetime_years": 30,
        "adoption_metric": "carbon_coverage_pct",
        "sdg_links": ["SDG13"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["gov_effectiveness > 0.5", "regulatory_quality > 0.4"],
        "estimated_effects": {
            "co2_per_capita_t": -0.8,
            "coal_share_pct": -5.0,
            "renewables_share_pct": 2.0,
        },
        "cost_category": "low",
        "description": "Implement or strengthen carbon pricing mechanisms."
    },
    {
        "tech_id": "PUBLIC_TRANSIT",
        "name": "Public Transit Investment",
        "family": "transport",
        "maturity_level": "mature",
        "lead_time_years": 4,
        "lifetime_years": 30,
        "adoption_metric": "public_transit_proxy",
        "sdg_links": ["SDG9", "SDG13"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["urban_pct > 50", "gov_effectiveness > 0.3"],
        "estimated_effects": {
            "co2_per_capita_t": -0.3,
            "energy_intensity_mj_usd": -0.2,
        },
        "cost_category": "high",
        "description": "Expand urban rail, BRT, and cycling infrastructure."
    },
    {
        "tech_id": "METHANE_CONTROL",
        "name": "Methane Leakage Control",
        "family": "emissions",
        "maturity_level": "mature",
        "lead_time_years": 1,
        "lifetime_years": 15,
        "adoption_metric": "methane_regulation_proxy",
        "sdg_links": ["SDG13"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["regulatory_quality > 0.3"],
        "estimated_effects": {
            "co2_per_capita_t": -0.3,
        },
        "cost_category": "low",
        "description": "Detect and fix methane leaks in oil/gas and agriculture."
    },
    {
        "tech_id": "RD_INNOVATION",
        "name": "R&D / Innovation Policy",
        "family": "innovation",
        "maturity_level": "policy",
        "lead_time_years": 3,
        "lifetime_years": 20,
        "adoption_metric": "rd_expenditure_gdp_pct",
        "sdg_links": ["SDG9"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["gdp_pc_ppp > 8000", "education_exp_gdp_pct > 3"],
        "estimated_effects": {
            "rd_expenditure_gdp_pct": 0.3,
            "energy_intensity_mj_usd": -0.2,
        },
        "cost_category": "medium",
        "description": "Public R&D funding + green innovation incentives."
    },
    {
        "tech_id": "INDUSTRIAL_EFFICIENCY",
        "name": "Industrial Efficiency Standards",
        "family": "efficiency",
        "maturity_level": "mature",
        "lead_time_years": 2,
        "lifetime_years": 15,
        "adoption_metric": "energy_intensity_mj_usd",
        "sdg_links": ["SDG9", "SDG13"],
        "prerequisites_hard": [],
        "prerequisites_soft": ["regulatory_quality > 0.3", "industry_gdp_pct > 15"],
        "estimated_effects": {
            "energy_intensity_mj_usd": -0.6,
            "co2_per_capita_t": -0.3,
            "manufacturing_va_gdp_pct": 0.5,
        },
        "cost_category": "medium",
        "description": "Mandatory efficiency standards for industrial processes."
    },
    {
        "tech_id": "STORAGE_BATTERY",
        "name": "Battery Storage Deployment",
        "family": "storage",
        "maturity_level": "scaling",
        "lead_time_years": 2,
        "lifetime_years": 15,
        "adoption_metric": "storage_capacity_proxy",
        "sdg_links": ["SDG7"],
        "prerequisites_hard": ["GRID_UPGRADE", "SOLAR_SCALE"],
        "prerequisites_soft": ["capability_index > 0.5"],
        "estimated_effects": {
            "renewables_share_pct": 3.0,
            "electricity_access_pct": 1.0,
        },
        "cost_category": "high",
        "description": "Grid-scale and distributed battery storage for RE integration."
    },
    {
        "tech_id": "EV_TRANSITION",
        "name": "EV Adoption Support",
        "family": "transport",
        "maturity_level": "scaling",
        "lead_time_years": 3,
        "lifetime_years": 15,
        "adoption_metric": "ev_share_proxy",
        "sdg_links": ["SDG13", "SDG9"],
        "prerequisites_hard": ["GRID_UPGRADE"],
        "prerequisites_soft": ["gdp_pc_ppp > 15000"],
        "estimated_effects": {
            "co2_per_capita_t": -0.4,
            "energy_intensity_mj_usd": -0.2,
        },
        "cost_category": "medium",
        "description": "EV purchase incentives, charging infrastructure, fleet mandates."
    },
]


def build_intervention_graph():
    """Build NetworkX DAG from intervention definitions."""
    G = nx.DiGraph()
    for intv in INTERVENTIONS:
        G.add_node(intv["tech_id"], **{k: v for k, v in intv.items() if k != "tech_id"})
    for intv in INTERVENTIONS:
        for prereq in intv["prerequisites_hard"]:
            G.add_edge(prereq, intv["tech_id"], type="hard")
    return G


def check_prerequisites(intervention, country_state):
    """Check if a country meets prerequisites for an intervention.
    Returns (feasible: bool, penalty: float, reasons: list)
    """
    intv = intervention
    reasons = []
    penalty = 0.0

    # Hard prerequisites (other interventions)
    for prereq_id in intv.get("prerequisites_hard", []):
        # In real system, check if prereq is already adopted
        # Here we check if the country has above-median adoption metrics
        pass  # Handled by graph traversal

    # Soft prerequisites
    for cond_str in intv.get("prerequisites_soft", []):
        # Parse "feature > threshold"
        parts = cond_str.split()
        if len(parts) == 3:
            feat, op, thresh = parts[0], parts[1], float(parts[2])
            val = country_state.get(feat, 0)
            if val is None or np.isnan(val):
                penalty += 0.2
                reasons.append(f"{feat}: data missing")
            elif op == ">" and val <= thresh:
                gap = (thresh - val) / max(thresh, 1)
                penalty += min(0.5, gap)
                reasons.append(f"{feat}={val:.2f} < {thresh}")

    feasible = penalty < 1.0
    return feasible, penalty, reasons


# ─── Utility function ────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "w_sdg": 0.25,       # Overall SDG improvement
    "w_co2": 0.30,       # CO₂ reduction
    "w_equity": 0.10,    # Inequality reduction
    "w_feasibility": 0.15,  # Capability constraint
    "w_speed": 0.10,     # Lead time (shorter = better)
    "w_cost": 0.10,      # Cost of capital / affordability
}


def score_intervention(intervention, country_state, models=None, feature_cols=None, weights=None):
    """Score a single intervention for a given country state.
    Returns utility score and component breakdown.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    effects = intervention["estimated_effects"]

    # SDG score: weighted sum of positive effects on SDG indicators
    sdg_score = 0
    sdg_score += effects.get("renewables_share_pct", 0) * 0.03
    sdg_score += effects.get("electricity_access_pct", 0) * 0.03
    sdg_score += effects.get("energy_intensity_mj_usd", 0) * (-0.04)  # lower is better
    sdg_score += effects.get("rd_expenditure_gdp_pct", 0) * 0.05

    # CO₂ score (reduction is positive)
    co2_score = -effects.get("co2_per_capita_t", 0)

    # Equity: currently proxied by how pro-poor the intervention is
    equity_score = 0.5  # neutral default
    if intervention["family"] in ["infrastructure", "transport"]:
        equity_score = 0.7

    # Feasibility
    _, penalty, _ = check_prerequisites(intervention, country_state)
    feasibility_score = max(0, 1 - penalty)

    # Speed (shorter lead time = higher score)
    speed_score = max(0, 1 - intervention["lead_time_years"] / 5)

    # Cost
    cost_map = {"low": 0.9, "medium": 0.5, "high": 0.2}
    cost_score = cost_map.get(intervention["cost_category"], 0.5)

    utility = (
        weights["w_sdg"] * sdg_score
        + weights["w_co2"] * co2_score
        + weights["w_equity"] * equity_score
        + weights["w_feasibility"] * feasibility_score
        + weights["w_speed"] * speed_score
        + weights["w_cost"] * cost_score
    )

    return {
        "tech_id": intervention["tech_id"],
        "name": intervention["name"],
        "utility": utility,
        "components": {
            "sdg": sdg_score,
            "co2": co2_score,
            "equity": equity_score,
            "feasibility": feasibility_score,
            "speed": speed_score,
            "cost": cost_score,
        },
        "penalty": penalty,
        "estimated_effects": effects,
        "lead_time": intervention["lead_time_years"],
    }


def generate_recommendations(country_iso3, country_state, weights=None, top_n=10):
    """Generate ranked intervention recommendations for a country.
    Respects hard prerequisites via topological ordering.
    """
    G = build_intervention_graph()

    # Score all interventions
    scored = []
    for intv in INTERVENTIONS:
        result = score_intervention(intv, country_state, weights=weights)
        # Check hard prerequisites
        prereqs = intv.get("prerequisites_hard", [])
        result["prerequisites_met"] = True  # Simplified; full version checks adoption
        result["prerequisites"] = prereqs
        scored.append(result)

    # Sort by utility
    scored.sort(key=lambda x: -x["utility"])

    # Build ordered roadmap respecting dependencies
    roadmap = []
    adopted = set()
    for item in scored[:top_n]:
        # Ensure prerequisites are included first
        for prereq in item.get("prerequisites", []):
            if prereq not in adopted:
                prereq_item = next((s for s in scored if s["tech_id"] == prereq), None)
                if prereq_item and prereq_item not in roadmap:
                    roadmap.append(prereq_item)
                    adopted.add(prereq)
        if item["tech_id"] not in adopted:
            roadmap.append(item)
            adopted.add(item["tech_id"])

    return {
        "country": country_iso3,
        "weights": weights or DEFAULT_WEIGHTS,
        "roadmap": roadmap,
        "n_interventions": len(roadmap),
    }


def run_recommendations(featured_df, output_dir, case_study_countries=("POL", "DEU", "IND")):
    """Generate recommendations for case study countries."""
    print("[PLANNER] Generating recommendations...")

    all_recs = {}
    for iso3 in case_study_countries:
        # Get latest available state
        country_data = featured_df[featured_df["country_iso3"] == iso3].sort_values("year")
        if country_data.empty:
            continue
        latest = country_data.iloc[-1].to_dict()

        recs = generate_recommendations(iso3, latest)
        all_recs[iso3] = recs

        # Save CSV
        roadmap_df = pd.DataFrame(recs["roadmap"])
        roadmap_df.to_csv(f"{output_dir}/{iso3}_roadmap.csv", index=False)
        print(f"  {iso3}: {len(recs['roadmap'])} interventions recommended")

    # Save full recommendations
    with open(f"{output_dir}/all_recommendations.json", "w") as f:
        json.dump(all_recs, f, indent=2, default=str)

    # Save intervention graph metadata
    with open(f"{output_dir}/intervention_graph.json", "w") as f:
        json.dump(INTERVENTIONS, f, indent=2)

    return all_recs
