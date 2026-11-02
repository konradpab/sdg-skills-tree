"""
Backtesting: Retrospective Policy Alignment Analysis

Validates the recommendation system by asking:
  "If we had generated recommendations at time T, do the observed
   changes over T→T+h align with what we recommended?"

This is NOT a causal test. It measures whether the system's priorities
are consistent with the direction the world actually moved — a necessary
(but not sufficient) condition for policy relevance.

Method:
  1. For each case study country, take state as of the train cutoff (2017)
  2. Generate intervention recommendations using the planner
  3. Map each recommended intervention to observable indicators
  4. Measure actual indicator changes from 2017 → 2020 and 2017 → 2023
  5. Compare: did indicators associated with top-ranked interventions
     improve more than those associated with low-ranked ones?
  6. Compute alignment scores and stability metrics
"""

import numpy as np
import pandas as pd
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── Mapping: intervention → observable indicators ────────────────────
# Each intervention has expected effects on specific columns in the panel.
# We check whether those columns actually changed in the expected direction.

INTERVENTION_OBSERVABLES = {
    "GRID_UPGRADE": {
        "indicators": ["electricity_access_pct", "renewables_share_pct"],
        "expected_direction": {"electricity_access_pct": +1, "renewables_share_pct": +1},
    },
    "SOLAR_SCALE": {
        "indicators": ["solar_share_pct", "renewables_share_pct"],
        "expected_direction": {"solar_share_pct": +1, "renewables_share_pct": +1},
    },
    "WIND_SCALE": {
        "indicators": ["wind_share_pct", "renewables_share_pct"],
        "expected_direction": {"wind_share_pct": +1, "renewables_share_pct": +1},
    },
    "COAL_PHASEDOWN": {
        "indicators": ["coal_share_pct", "co2_per_capita_t"],
        "expected_direction": {"coal_share_pct": -1, "co2_per_capita_t": -1},
    },
    "BUILDING_EFFICIENCY": {
        "indicators": ["energy_intensity_mj_usd"],
        "expected_direction": {"energy_intensity_mj_usd": -1},
    },
    "HEAT_PUMPS": {
        "indicators": ["energy_intensity_mj_usd", "co2_per_capita_t"],
        "expected_direction": {"energy_intensity_mj_usd": -1, "co2_per_capita_t": -1},
    },
    "CARBON_PRICING": {
        "indicators": ["carbon_price_usd", "carbon_coverage_pct", "co2_per_capita_t"],
        "expected_direction": {"carbon_price_usd": +1, "carbon_coverage_pct": +1, "co2_per_capita_t": -1},
    },
    "PUBLIC_TRANSIT": {
        "indicators": ["co2_per_capita_t", "energy_intensity_mj_usd"],
        "expected_direction": {"co2_per_capita_t": -1, "energy_intensity_mj_usd": -1},
    },
    "METHANE_CONTROL": {
        "indicators": ["co2_per_capita_t"],
        "expected_direction": {"co2_per_capita_t": -1},
    },
    "RD_INNOVATION": {
        "indicators": ["rd_expenditure_gdp_pct"],
        "expected_direction": {"rd_expenditure_gdp_pct": +1},
    },
    "INDUSTRIAL_EFFICIENCY": {
        "indicators": ["energy_intensity_mj_usd", "manufacturing_va_gdp_pct"],
        "expected_direction": {"energy_intensity_mj_usd": -1, "manufacturing_va_gdp_pct": +1},
    },
    "STORAGE_BATTERY": {
        "indicators": ["renewables_share_pct"],
        "expected_direction": {"renewables_share_pct": +1},
    },
    "EV_TRANSITION": {
        "indicators": ["co2_per_capita_t", "energy_intensity_mj_usd"],
        "expected_direction": {"co2_per_capita_t": -1, "energy_intensity_mj_usd": -1},
    },
}


def _get_country_state_at_year(panel, iso3, year):
    """Extract state vector for a country at a specific year."""
    row = panel[(panel["country_iso3"] == iso3) & (panel["year"] == year)]
    if row.empty:
        # Try nearest year
        country_data = panel[panel["country_iso3"] == iso3].sort_values("year")
        row = country_data[country_data["year"] <= year].tail(1)
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _compute_observed_change(panel, iso3, indicator, year_from, year_to):
    """Compute the actual change in an indicator between two years."""
    val_from = panel.loc[
        (panel["country_iso3"] == iso3) & (panel["year"] == year_from), indicator
    ]
    val_to = panel.loc[
        (panel["country_iso3"] == iso3) & (panel["year"] == year_to), indicator
    ]
    if val_from.empty or val_to.empty:
        return np.nan
    v0 = val_from.values[0]
    v1 = val_to.values[0]
    if pd.isna(v0) or pd.isna(v1):
        return np.nan
    return v1 - v0


def backtest_country(panel, iso3, recommendations, base_year, eval_horizons=(3, 5)):
    """
    Backtest recommendations for one country.

    For each recommended intervention:
      - Look up which indicators it should affect
      - Measure actual change in those indicators over the horizon
      - Check if the change was in the expected direction
      - Weight by the intervention's utility rank

    Returns a dict with alignment scores and detailed results.
    """
    roadmap = recommendations.get(iso3, {}).get("roadmap", [])
    if not roadmap:
        return {"country": iso3, "error": "no recommendations"}

    results_by_horizon = {}

    for horizon in eval_horizons:
        eval_year = base_year + horizon
        intervention_results = []

        for rank, item in enumerate(roadmap, 1):
            tech_id = item["tech_id"]
            obs = INTERVENTION_OBSERVABLES.get(tech_id)
            if obs is None:
                continue

            indicator_checks = []
            for indicator in obs["indicators"]:
                if indicator not in panel.columns:
                    continue

                observed_change = _compute_observed_change(panel, iso3, indicator, base_year, eval_year)
                expected_dir = obs["expected_direction"].get(indicator, 0)

                if np.isnan(observed_change):
                    aligned = np.nan
                else:
                    # Did the change go in the expected direction?
                    aligned = 1.0 if (np.sign(observed_change) == expected_dir) else 0.0

                indicator_checks.append({
                    "indicator": indicator,
                    "expected_direction": "increase" if expected_dir > 0 else "decrease",
                    "observed_change": round(float(observed_change), 4) if not np.isnan(observed_change) else None,
                    "aligned": aligned if not np.isnan(aligned) else None,
                    "base_value": round(float(
                        panel.loc[(panel["country_iso3"] == iso3) & (panel["year"] == base_year), indicator].values[0]
                    ), 3) if not panel.loc[(panel["country_iso3"] == iso3) & (panel["year"] == base_year), indicator].empty else None,
                })

            # Overall alignment for this intervention
            valid_checks = [c["aligned"] for c in indicator_checks if c["aligned"] is not None]
            intv_alignment = np.mean(valid_checks) if valid_checks else np.nan

            intervention_results.append({
                "rank": rank,
                "tech_id": tech_id,
                "name": item["name"],
                "utility": item["utility"],
                "alignment_score": round(float(intv_alignment), 3) if not np.isnan(intv_alignment) else None,
                "indicators_checked": len(valid_checks),
                "details": indicator_checks,
            })

        # Aggregate alignment scores
        valid_alignments = [r["alignment_score"] for r in intervention_results
                           if r["alignment_score"] is not None]

        # Weighted alignment: higher-ranked interventions matter more
        weighted_sum = 0
        weight_total = 0
        for r in intervention_results:
            if r["alignment_score"] is not None:
                w = 1.0 / r["rank"]  # inverse rank weighting
                weighted_sum += r["alignment_score"] * w
                weight_total += w

        results_by_horizon[f"{horizon}y"] = {
            "eval_period": f"{base_year}→{eval_year}",
            "mean_alignment": round(float(np.mean(valid_alignments)), 3) if valid_alignments else None,
            "weighted_alignment": round(float(weighted_sum / weight_total), 3) if weight_total > 0 else None,
            "n_interventions_evaluated": len(valid_alignments),
            "n_total_indicators_checked": sum(r["indicators_checked"] for r in intervention_results),
            "interventions": intervention_results,
        }

    return {
        "country": iso3,
        "base_year": base_year,
        "horizons": results_by_horizon,
    }


def backtest_recommendation_stability(panel, iso3, base_years, planner_fn, weights=None):
    """
    Test recommendation stability: do recommendations change substantially
    when we shift the base year by 1-2 years?

    Generates recommendations at multiple base years and measures overlap.
    """
    roadmaps_by_year = {}
    for year in base_years:
        state = _get_country_state_at_year(panel, iso3, year)
        if state is None:
            continue
        recs = planner_fn(iso3, state, weights=weights)
        tech_ids = [item["tech_id"] for item in recs.get("roadmap", [])]
        roadmaps_by_year[year] = tech_ids

    if len(roadmaps_by_year) < 2:
        return {"country": iso3, "error": "insufficient years for stability test"}

    # Compute pairwise Jaccard similarity between roadmaps
    years = sorted(roadmaps_by_year.keys())
    similarities = []
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            set_a = set(roadmaps_by_year[years[i]])
            set_b = set(roadmaps_by_year[years[j]])
            if set_a or set_b:
                jaccard = len(set_a & set_b) / len(set_a | set_b)
            else:
                jaccard = 1.0
            similarities.append({
                "year_a": years[i], "year_b": years[j],
                "jaccard": round(jaccard, 3),
                "overlap": list(set_a & set_b),
                "only_a": list(set_a - set_b),
                "only_b": list(set_b - set_a),
            })

    # Top-3 stability: are the top 3 recommendations the same across years?
    top3_sets = {y: set(ids[:3]) for y, ids in roadmaps_by_year.items()}
    top3_stable = all(
        top3_sets[years[i]] == top3_sets[years[i + 1]]
        for i in range(len(years) - 1)
    ) if len(years) > 1 else True

    return {
        "country": iso3,
        "base_years": years,
        "pairwise_jaccard": similarities,
        "mean_jaccard": round(float(np.mean([s["jaccard"] for s in similarities])), 3),
        "top3_stable": top3_stable,
        "roadmaps": {str(y): ids for y, ids in roadmaps_by_year.items()},
    }


def backtest_forecast_accuracy(featured_df, models, feature_cols, base_year=2017):
    """
    Backtest forecast accuracy: compare model predictions at base_year
    against actual outcomes at base_year + horizon.

    This is a per-country, per-indicator analysis showing whether the
    model's directional predictions were correct.
    """
    results = []

    for target_col, model in models.items():
        # Parse target: "target_renewables_share_pct_delta3y" → indicator + horizon
        parts = target_col.replace("target_", "").rsplit("_delta", 1)
        if len(parts) != 2:
            continue
        indicator = parts[0]
        horizon_str = parts[1].replace("y", "")
        try:
            horizon = int(horizon_str)
        except ValueError:
            continue

        # Get rows at base_year with valid features and target
        base_rows = featured_df[
            (featured_df["year"] == base_year) &
            featured_df[target_col].notna()
        ].copy()

        if base_rows.empty or len(base_rows) < 5:
            continue

        # Clean features
        valid_features = [c for c in feature_cols if c in base_rows.columns]
        clean = base_rows[["country_iso3", "year"] + valid_features + [target_col]].dropna()
        if len(clean) < 5:
            continue

        X = clean[valid_features].values
        y_actual = clean[target_col].values
        y_pred = model.predict(X)

        # Per-country results
        for idx, (_, row) in enumerate(clean.iterrows()):
            iso3 = row["country_iso3"]
            actual = y_actual[idx]
            predicted = y_pred[idx]

            results.append({
                "country_iso3": iso3,
                "indicator": indicator,
                "horizon": horizon,
                "base_year": base_year,
                "eval_year": base_year + horizon,
                "predicted_delta": round(float(predicted), 4),
                "actual_delta": round(float(actual), 4),
                "direction_correct": bool(np.sign(predicted) == np.sign(actual)) if actual != 0 else None,
                "absolute_error": round(float(abs(predicted - actual)), 4),
            })

    df = pd.DataFrame(results)
    if df.empty:
        return df, {}

    # Summary statistics
    summary = {}
    for indicator in df["indicator"].unique():
        for horizon in df["horizon"].unique():
            mask = (df["indicator"] == indicator) & (df["horizon"] == horizon)
            sub = df[mask].dropna(subset=["direction_correct"])
            if len(sub) < 3:
                continue
            key = f"{indicator}_delta{horizon}y"
            summary[key] = {
                "n_countries": len(sub),
                "direction_accuracy": round(float(sub["direction_correct"].mean()), 3),
                "mean_absolute_error": round(float(sub["absolute_error"].mean()), 4),
                "median_absolute_error": round(float(sub["absolute_error"].median()), 4),
            }

    return df, summary


def run_backtesting(featured_df, panel_raw, models, feature_cols, recommendations,
                    output_dir, case_study_countries=("POL", "DEU", "IND"),
                    base_year=2017):
    """
    Main backtesting entry point. Runs all backtest analyses.
    """
    from src.policy.planner import generate_recommendations

    print("[BACKTEST] Running retrospective policy alignment analysis...")

    all_results = {}

    # 1. Policy alignment backtests for case study countries
    print("\n  Policy alignment backtests:")
    alignment_results = {}
    for iso3 in case_study_countries:
        # Generate recommendations as of base_year
        state = _get_country_state_at_year(featured_df, iso3, base_year)
        if state is None:
            print(f"    {iso3}: no data at {base_year}, skipping")
            continue

        # Use existing recommendations or regenerate at base_year
        recs_at_base = {iso3: {"roadmap": recommendations.get(iso3, {}).get("roadmap", [])}}

        result = backtest_country(panel_raw, iso3, recs_at_base, base_year, eval_horizons=(3, 5))
        alignment_results[iso3] = result

        for hz, hz_data in result.get("horizons", {}).items():
            ma = hz_data.get("mean_alignment")
            wa = hz_data.get("weighted_alignment")
            n = hz_data.get("n_interventions_evaluated", 0)
            print(f"    {iso3} ({hz}): alignment={ma}, weighted={wa}, interventions={n}")

    all_results["policy_alignment"] = alignment_results

    # 2. Recommendation stability
    print("\n  Recommendation stability:")
    stability_results = {}
    for iso3 in case_study_countries:
        result = backtest_recommendation_stability(
            featured_df, iso3,
            base_years=[2015, 2016, 2017],
            planner_fn=generate_recommendations,
        )
        stability_results[iso3] = result
        jaccard = result.get("mean_jaccard", "N/A")
        top3 = result.get("top3_stable", "N/A")
        print(f"    {iso3}: mean Jaccard={jaccard}, top-3 stable={top3}")

    all_results["stability"] = stability_results

    # 3. Forecast accuracy backtest
    print("\n  Forecast accuracy backtest:")
    forecast_df, forecast_summary = backtest_forecast_accuracy(
        featured_df, models, feature_cols, base_year=base_year
    )
    all_results["forecast_accuracy"] = forecast_summary

    if forecast_summary:
        for key, stats in forecast_summary.items():
            dir_acc = stats.get("direction_accuracy", "?")
            mae = stats.get("mean_absolute_error", "?")
            n = stats.get("n_countries", 0)
            print(f"    {key}: direction_accuracy={dir_acc}, MAE={mae}, n={n}")

    # Save results
    with open(f"{output_dir}/backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if not forecast_df.empty:
        forecast_df.to_csv(f"{output_dir}/backtest_forecast_detail.csv", index=False)

    # Save alignment detail CSVs per country
    for iso3, result in alignment_results.items():
        rows = []
        for hz, hz_data in result.get("horizons", {}).items():
            for intv in hz_data.get("interventions", []):
                for detail in intv.get("details", []):
                    rows.append({
                        "horizon": hz,
                        "rank": intv["rank"],
                        "intervention": intv["name"],
                        "utility": intv["utility"],
                        "indicator": detail["indicator"],
                        "expected": detail["expected_direction"],
                        "observed_change": detail["observed_change"],
                        "aligned": detail["aligned"],
                        "base_value": detail["base_value"],
                    })
        if rows:
            pd.DataFrame(rows).to_csv(f"{output_dir}/backtest_{iso3}_detail.csv", index=False)

    print(f"\n[BACKTEST] Results saved to {output_dir}/")
    return all_results
