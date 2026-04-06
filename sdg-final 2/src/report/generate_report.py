"""
Generate the final HTML report using Jinja2.
Replaces Quarto since it is not available in this environment.
"""

import json
import os
import base64
from datetime import datetime, timezone
from jinja2 import Template
import pandas as pd
import numpy as np


def _img_to_base64(path):
    """Embed image as base64 data URI."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _format_metrics_table(results, test_metrics):
    """Format model metrics as HTML table."""
    rows = []
    for target, info in results.items():
        label = info.get("label", target)
        best = info.get("best_model", "")
        for model, m in info["metrics"].items():
            is_best = "★" if model == best else ""
            test_m = test_metrics.get(target, {})
            rows.append({
                "Target": label[:35],
                "Model": f"{model} {is_best}",
                "Train MAE": f"{m['train_mae']:.4f}",
                "Val MAE": f"{m['val_mae']:.4f}",
                "Val RMSE": f"{m['val_rmse']:.4f}",
                "Val Rank ρ": f"{m.get('val_rank_corr', 0):.3f}",
                "Test MAE": f"{test_m.get('test_mae', '-')}" if model == best else "-",
                "Conformal Cov.": f"{test_m.get('conformal_coverage_90pct', '-')}" if model == best else "-",
            })
    return rows


def _format_recommendation_table(recs, iso3):
    """Format recommendations as HTML table."""
    roadmap = recs.get(iso3, {}).get("roadmap", [])
    rows = []
    for i, item in enumerate(roadmap, 1):
        rows.append({
            "Rank": i,
            "Intervention": item["name"],
            "Utility": f"{item['utility']:.3f}",
            "CO₂ Effect": f"{item.get('estimated_effects', {}).get('co2_per_capita_t', 0):.2f} t/cap",
            "RE Effect": f"{item.get('estimated_effects', {}).get('renewables_share_pct', 0):.1f} pp",
            "Lead Time": f"{item.get('lead_time', '?')}y",
            "Feasibility": f"{item['components']['feasibility']:.2f}",
            "Prerequisites": ", ".join(item.get("prerequisites", [])) or "None",
        })
    return rows


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  :root { --accent: #2C5F2D; --accent2: #DA7B27; --bg: #fafaf8; --text: #1a1a1a; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Georgia', 'Times New Roman', serif; line-height: 1.7; color: var(--text);
         background: var(--bg); max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 2rem; color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: 0.5rem; margin: 2rem 0 1rem; }
  h2 { font-size: 1.5rem; color: var(--accent); margin: 2rem 0 0.8rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
  h3 { font-size: 1.2rem; color: var(--accent2); margin: 1.5rem 0 0.5rem; }
  p { margin: 0.6rem 0; text-align: justify; }
  img { max-width: 100%; height: auto; margin: 1rem 0; border: 1px solid #e0e0e0; border-radius: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; font-family: 'Helvetica', sans-serif; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
  th { background: var(--accent); color: white; font-weight: 600; }
  tr:nth-child(even) { background: #f5f5f0; }
  .meta { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
  .caveat { background: #fff3cd; border-left: 4px solid var(--accent2); padding: 0.8rem 1rem; margin: 1rem 0; font-size: 0.9rem; }
  .section-nav { background: #f0f0e8; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
  .section-nav a { color: var(--accent); text-decoration: none; margin-right: 1rem; }
  code { background: #f0f0e0; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  .figure-caption { text-align: center; font-style: italic; color: #666; font-size: 0.9rem; margin-top: -0.5rem; }
  .toc { background: #f8f8f3; padding: 1.5rem; border: 1px solid #ddd; margin: 1rem 0; }
  .toc a { display: block; padding: 3px 0; color: var(--accent); text-decoration: none; }
  @media print { body { max-width: 100%; } img { max-width: 100%; } }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p class="meta">
  Generated: {{ timestamp }} | Countries: {{ n_countries }} | Years: {{ year_range }} |
  Pipeline: {{ data_mode }}
</p>

<div class="toc">
<strong>Contents</strong>
<a href="#motivation">1. Motivation & Policy Context</a>
<a href="#data">2. Data & Provenance</a>
<a href="#methods">3. Methods</a>
<a href="#results">4. Results</a>
<a href="#casestudies">5. Case Studies</a>
<a href="#validation">6. Validation & Backtesting</a>
<a href="#discussion">7. Discussion</a>
<a href="#conclusion">8. Conclusion</a>
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="motivation">1. Motivation & Policy Context</h2>

<p>The EU Green Deal, Fit for 55 package, and the global SDG framework demand accelerated
decarbonization while safeguarding equity and development. Yet policy-makers face a combinatorial
challenge: which interventions to prioritize, in what sequence, given heterogeneous national
capabilities, institutions, and energy systems.</p>

<p>This system addresses that challenge by building a data-driven, interpretable pipeline that
learns region-specific technology and policy adoption pathways—"skill-trees"—that maximize
SDG progress and emissions reduction while respecting feasibility constraints. The pipeline
ingests open data on outcomes, capabilities, governance, inequality, and policy proxies to
produce ranked roadmaps with uncertainty quantification.</p>

{% if is_synthetic %}
<div class="caveat">
<strong>Note:</strong> This report was generated using <em>synthetic data</em> that mirrors
the structure and distributions of real-world datasets (see Section 2). All model results
are illustrative of the pipeline's capabilities. With real data, the same code produces
policy-grade analysis.
</div>
{% else %}
<div class="caveat">
<strong>Note:</strong> This report uses observational data from {{ n_datasets }} open
international datasets. Model results reflect statistical associations, not causal effects.
Intervention effect estimates are proxied from observational variation; see Section 6 for
limitations and caveats.
</div>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="data">2. Data & Provenance</h2>

<h3>2.1 Dataset Stack</h3>
<p>The Minimum Viable Stack (MVS) draws on {{ n_datasets }} curated open datasets spanning
outcomes (SDG indicators, emissions, electricity mix), capabilities (governance, institutions,
education), inequality (SWIID Gini), policy proxies (carbon pricing), and risk (conflict events).
Each dataset is tracked in a machine-readable registry with source URLs, licenses, retrieval
methods, version hashes, and BibTeX citations.</p>

<table>
<tr><th>ID</th><th>Name</th><th>Domain</th><th>License</th><th>Fields Used</th></tr>
{% for ds in datasets %}
<tr>
  <td><code>{{ ds.id }}</code></td>
  <td>{{ ds.name }}</td>
  <td>{{ ds.domain }}</td>
  <td>{{ ds.license }}</td>
  <td>{{ ds.fields_used | length }} fields</td>
</tr>
{% endfor %}
</table>

<h3>2.2 Data Coverage</h3>
{% if img_coverage %}
<img src="{{ img_coverage }}" alt="Data coverage heatmap">
<p class="figure-caption">Figure 1: Data coverage by country and indicator. Green = complete, yellow = partial.</p>
{% endif %}

<h3>2.3 Unified Schema</h3>
<p>All datasets are normalized to a canonical <code>country_iso3 × year</code> grid.
The master panel contains {{ n_rows }} observations across {{ n_features }} features
for {{ n_countries }} countries over {{ year_range }}.</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="methods">3. Methods</h2>

<h3>3.1 Feature Engineering</h3>
<p>From raw indicators we construct: lagged values (t-1, t-2), backward deltas (Δ1y, Δ3y),
3-year rolling means, and three composite indices—<em>capability</em> (governance + GDP +
education + R&D), <em>feasibility</em> (capability + stability − conflict), and
<em>green transition</em> (renewables share + coal exit + energy intensity + carbon pricing coverage).
Log transforms are applied to skewed variables (GDP, population, generation, conflict events).</p>

<h3>3.2 Forecasting Models</h3>
<p>We train five model families per target variable with time-aware validation
(train ≤ 2017, validation 2018–2020, test 2021+):</p>
<p><strong>Baselines:</strong> naive persistence (Δ=0), elastic net.
<strong>Core:</strong> random forest, gradient boosting (GBR), histogram gradient boosting (HistGBR).
All use consistent feature sets for comparability.</p>

<h3>3.3 Uncertainty Estimation</h3>
<p>Two complementary approaches: (1) <em>quantile regression</em> via GBR with pinball loss
at quantiles {0.10, 0.25, 0.50, 0.75, 0.90}, producing prediction fan charts; and
(2) <em>split conformal prediction</em> calibrated on the validation set, yielding
distribution-free coverage guarantees at the 90% level.</p>

<h3>3.4 Intervention Graph & Planner</h3>
<p>{{ n_interventions }} interventions are organized in a directed acyclic graph (DAG)
with hard and soft prerequisites. Each node specifies lead time, lifetime, cost category,
SDG links, and estimated effects. The planner scores each intervention via a multi-objective
utility function:</p>
<p><code>U = w_sdg·S_sdg + w_co2·S_co2 + w_equity·S_eq + w_feasibility·S_feas + w_speed·S_speed + w_cost·S_cost</code></p>
<p>Default weights: {{ weights }}. Interventions are ranked by utility and
sequenced to respect dependency constraints via topological ordering.</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="results">4. Results</h2>

<h3>4.1 Forecasting Performance</h3>
{% if img_model_comparison %}
<img src="{{ img_model_comparison }}" alt="Model comparison">
<p class="figure-caption">Figure 2: Validation MAE by model and target. ★ = best model selected.</p>
{% endif %}

<table>
<tr><th>Target</th><th>Model</th><th>Train MAE</th><th>Val MAE</th><th>Val RMSE</th><th>Val ρ</th><th>Test MAE</th><th>Conformal Cov.</th></tr>
{% for row in metrics_table %}
<tr>
  <td>{{ row['Target'] }}</td><td>{{ row['Model'] }}</td>
  <td>{{ row['Train MAE'] }}</td><td>{{ row['Val MAE'] }}</td>
  <td>{{ row['Val RMSE'] }}</td><td>{{ row['Val Rank ρ'] }}</td>
  <td>{{ row['Test MAE'] }}</td><td>{{ row['Conformal Cov.'] }}</td>
</tr>
{% endfor %}
</table>

<h3>4.2 Feature Importance</h3>
{% if img_feature_importance %}
<img src="{{ img_feature_importance }}" alt="Feature importance">
<p class="figure-caption">Figure 3: Top features by impurity-based importance (gradient boosting).</p>
{% endif %}

<h3>4.3 Indicator Trajectories</h3>
{% if img_traj_renewables %}
<img src="{{ img_traj_renewables }}" alt="Renewables trajectory">
<p class="figure-caption">Figure 4: Renewables share trajectories for case study countries.</p>
{% endif %}
{% if img_traj_co2 %}
<img src="{{ img_traj_co2 }}" alt="CO2 trajectory">
<p class="figure-caption">Figure 5: CO₂ per capita trajectories for case study countries.</p>
{% endif %}
{% if img_traj_ei %}
<img src="{{ img_traj_ei }}" alt="Energy intensity trajectory">
<p class="figure-caption">Figure 6: Energy intensity trajectories for case study countries.</p>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="casestudies">5. Case Studies</h2>

{% for iso3 in case_countries %}
<h3>5.{{ loop.index }} {{ iso3 }} — {{ country_names[iso3] }}</h3>

<p><strong>Current state (latest year):</strong></p>
<table>
<tr><th>Indicator</th><th>Value</th></tr>
{% for key, val in snapshots[iso3].items() %}
<tr><td>{{ key }}</td><td>{{ val }}</td></tr>
{% endfor %}
</table>

<p><strong>Recommended intervention roadmap:</strong></p>
<table>
<tr><th>#</th><th>Intervention</th><th>Utility</th><th>CO₂ Effect</th><th>RE Effect</th><th>Lead</th><th>Feasibility</th><th>Prerequisites</th></tr>
{% for row in rec_tables[iso3] %}
<tr>
  <td>{{ row['Rank'] }}</td><td>{{ row['Intervention'] }}</td>
  <td>{{ row['Utility'] }}</td><td>{{ row['CO₂ Effect'] }}</td>
  <td>{{ row['RE Effect'] }}</td><td>{{ row['Lead Time'] }}</td>
  <td>{{ row['Feasibility'] }}</td><td>{{ row['Prerequisites'] }}</td>
</tr>
{% endfor %}
</table>

{% if roadmap_imgs[iso3] %}
<img src="{{ roadmap_imgs[iso3] }}" alt="Roadmap {{ iso3 }}">
<p class="figure-caption">Figure: Intervention roadmap for {{ iso3 }}. Light bar = lead time; dark bar = active lifetime.</p>
{% endif %}
{% endfor %}

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="validation">6. Validation & Backtesting</h2>

<h3>6.1 Retrospective Policy Alignment</h3>
<p>To validate the planner, we generate recommendations as of {{ backtest_base_year }} and check
whether the associated indicators actually moved in the expected direction during
{{ backtest_base_year }}→{{ backtest_base_year|int + 3 }} and {{ backtest_base_year }}→{{ backtest_base_year|int + 5 }}.
A weighted alignment score >0.5 indicates better-than-random consistency between
recommendations and observed outcomes.</p>

{% if img_backtest_alignment %}
<img src="{{ img_backtest_alignment }}" alt="Policy alignment">
<p class="figure-caption">Figure: Retrospective policy alignment scores by country and evaluation horizon.</p>
{% endif %}

<h3>6.2 Forecast Directional Accuracy</h3>
<p>For each forecasting target, we measure what fraction of per-country predictions had
the correct sign (positive or negative change). Accuracy above 50% indicates the model
captures directional trends beyond random chance.</p>

{% if img_backtest_forecast %}
<img src="{{ img_backtest_forecast }}" alt="Forecast accuracy">
<p class="figure-caption">Figure: Forecast directional accuracy by target indicator.</p>
{% endif %}

<h3>6.3 Recommendation Stability</h3>
<p>We test whether recommendations are stable when the base year shifts by 1–2 years
(2015 vs 2016 vs 2017). High Jaccard similarity indicates robust recommendations that
are not artifacts of particular data configurations.</p>

{% if img_backtest_stability %}
<img src="{{ img_backtest_stability }}" alt="Recommendation stability">
<p class="figure-caption">Figure: Jaccard similarity of recommended intervention sets across base years.</p>
{% endif %}

<h3>6.4 Ablation: Feature Group Contribution</h3>
<p>We remove governance features and cultural features separately to measure their
marginal contribution. A positive percentage indicates that removing the feature group
<em>increased</em> MAE (i.e. the features helped). Negative means they added noise.</p>

{% if img_ablation %}
<img src="{{ img_ablation }}" alt="Ablation results">
<p class="figure-caption">Figure: Ablation study — change in validation MAE when feature groups are removed.</p>
{% endif %}

<h3>6.5 SHAP Feature Attribution</h3>
{% if img_shap %}
<img src="{{ img_shap }}" alt="SHAP importance">
<p class="figure-caption">Figure: SHAP-based feature importance (mean |SHAP value|) for key forecast targets.</p>
{% endif %}

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="discussion">7. Discussion</h2>

<h3>7.1 Limitations</h3>
<p><strong>Causality:</strong> This system produces counterfactual simulations, not causal estimates.
Intervention effects are proxied via observational associations with explicit uncertainty bounds.
Users should interpret recommendations as informed suggestions rather than guaranteed outcomes.</p>

<p><strong>Data quality:</strong> Synthetic data was used for this demonstration. Real-world
deployment requires careful handling of measurement error, temporal gaps, and cross-country
comparability issues inherent in international statistics.</p>

<p><strong>Culture variables:</strong> Cultural dimensions (Hofstede, Schwartz, WVS) are
treated as contextual heterogeneity features, never as deterministic constraints. The system
explicitly avoids encoding stereotypes or implying cultural determinism.</p>

<h3>7.2 Fairness & Ethics</h3>
<p>The feasibility constraints may systematically disadvantage low-income countries by penalizing
weak governance or low GDP. This is a known tension: removing these constraints produces
unrealistic recommendations, while including them may entrench existing inequalities. The
normative utility weights are made explicit and adjustable precisely so that users can
examine these trade-offs.</p>

<h3>7.3 Policy Implications</h3>
<p>The skill-tree framing highlights that interventions are not independent: grid upgrades
enable renewable scale-up, which enables storage deployment and EV adoption. Sequencing
matters. The system's dependency graph makes these relationships explicit and actionable
for policy planning.</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 id="conclusion">8. Conclusion</h2>

<p>This report demonstrates an end-to-end, reproducible ML pipeline that translates open
international data into country-specific intervention roadmaps for SDG progress and
decarbonization. The system combines gradient boosting forecasting with quantile and
conformal uncertainty, an intervention dependency graph, and a multi-objective planner.
Key contributions include the skill-tree formalization of policy sequencing, the explicit
treatment of feasibility constraints, and the fully auditable data-to-recommendation pipeline.</p>

<p>Next steps include: integration with real datasets via the documented API endpoints,
extension to EU NUTS2 regional granularity, incorporation of causal forest ablations,
and validation against observed policy outcomes.</p>

<hr style="margin-top: 3rem;">
<p class="meta">Pipeline version: 1.0 | Report generated: {{ timestamp }}</p>

</body>
</html>
"""


def generate_report(panel, results, test_metrics, importances, recs,
                    interventions, datasets, fig_dir, output_path,
                    case_countries=None):
    """Generate the full HTML report."""
    print("[REPORT] Generating HTML report...")

    template = Template(REPORT_TEMPLATE)

    # Use provided case countries or detect from recommendations
    if case_countries is None:
        case_countries = list(recs.keys()) if recs else ["POL", "DEU", "IND"]

    # Build country name lookup from panel or fallback
    all_iso3 = panel["country_iso3"].unique().tolist()
    country_names = {iso3: iso3 for iso3 in case_countries}  # fallback
    # Try to get real names from dim_country if available
    try:
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dim_path = os.path.join(root, "data_processed", "dim_country.csv")
        if os.path.exists(dim_path):
            dim = pd.read_csv(dim_path)
            if "country_name" in dim.columns and "country_iso3" in dim.columns:
                name_map = dict(zip(dim["country_iso3"], dim["country_name"]))
                country_names.update({iso3: name_map.get(iso3, iso3) for iso3 in case_countries})
    except Exception:
        pass
    # Common names fallback
    _NAMES = {"POL": "Poland", "DEU": "Germany", "IND": "India", "BRA": "Brazil",
              "ZAF": "South Africa", "NGA": "Nigeria", "NLD": "Netherlands",
              "FRA": "France", "GBR": "United Kingdom", "USA": "United States",
              "CHN": "China", "KEN": "Kenya", "CHL": "Chile", "MEX": "Mexico",
              "IDN": "Indonesia", "TUR": "Turkey", "EGY": "Egypt", "ETH": "Ethiopia"}
    for iso3 in case_countries:
        if country_names.get(iso3) == iso3 and iso3 in _NAMES:
            country_names[iso3] = _NAMES[iso3]

    # Snapshots
    snapshots = {}
    snapshot_cols = [
        "gdp_pc_ppp", "gov_effectiveness", "renewables_share_pct",
        "co2_per_capita_t", "energy_intensity_mj_usd", "gini_disposable",
        "carbon_price_usd", "capability_index", "feasibility_index",
        "green_transition_index"
    ]
    for iso3 in case_countries:
        cdata = panel[panel["country_iso3"] == iso3].sort_values("year")
        if cdata.empty:
            snapshots[iso3] = {}
            continue
        latest = cdata.iloc[-1]
        snap = {}
        for col in snapshot_cols:
            if col in latest.index:
                val = latest[col]
                if pd.notna(val):
                    snap[col.replace("_", " ").title()] = f"{val:.2f}"
        snapshots[iso3] = snap

    # Format tables
    metrics_table = _format_metrics_table(results, test_metrics)
    rec_tables = {iso3: _format_recommendation_table(recs, iso3) for iso3 in case_countries}

    # Embed images
    img = lambda name: _img_to_base64(f"{fig_dir}/{name}")
    roadmap_imgs = {iso3: img(f"roadmap_{iso3}.png") for iso3 in case_countries}

    # Weights string
    from src.policy.planner import DEFAULT_WEIGHTS
    weights_str = ", ".join(f"{k}={v}" for k, v in DEFAULT_WEIGHTS.items())

    # Feature count
    n_features = len([c for c in panel.columns if c not in ("country_iso3", "year")])

    # Detect if data is synthetic (50 countries) or real (typically 100+)
    n_countries = panel["country_iso3"].nunique()
    is_synthetic = n_countries <= 55
    data_mode = "synthetic data demonstration" if is_synthetic else f"real data ({n_countries} countries)"

    html = template.render(
        title="From Indicators to Interventions: Learning Personalized Policy-Tech Pathways Toward SDG Progress",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_countries=n_countries,
        is_synthetic=is_synthetic,
        data_mode=data_mode,
        year_range=f"{panel['year'].min()}–{panel['year'].max()}",
        n_datasets=len(datasets),
        datasets=datasets,
        n_rows=f"{len(panel):,}",
        n_features=n_features,
        n_interventions=len(interventions),
        weights=weights_str,
        metrics_table=metrics_table,
        img_coverage=img("data_coverage.png"),
        img_model_comparison=img("model_comparison.png"),
        img_feature_importance=img("feature_importance.png"),
        img_traj_renewables=img("trajectory_renewables_share_pct.png"),
        img_traj_co2=img("trajectory_co2_per_capita_t.png"),
        img_traj_ei=img("trajectory_energy_intensity_mj_usd.png"),
        case_countries=case_countries,
        country_names=country_names,
        snapshots=snapshots,
        rec_tables=rec_tables,
        roadmap_imgs=roadmap_imgs,
        # Backtesting & validation figures
        backtest_base_year=2017,
        img_backtest_alignment=img("backtest_alignment.png"),
        img_backtest_forecast=img("backtest_forecast_accuracy.png"),
        img_backtest_stability=img("backtest_stability.png"),
        img_ablation=img("ablation_study.png"),
        img_shap=img("shap_importance.png"),
    )

    with open(output_path, "w") as f:
        f.write(html)

    print(f"[REPORT] Saved to {output_path}")
    return output_path
