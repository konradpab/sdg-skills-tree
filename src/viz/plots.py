"""
Visualization module. Generates all figures for the report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#2C5F2D", "#97BC62", "#DA7B27", "#2B4162", "#C14953", "#7B9EA8", "#E8C547"]


def plot_data_coverage(panel, save_path):
    """Heatmap of data coverage (missingness) by country and variable."""
    key_cols = [
        "gdp_pc_ppp", "gov_effectiveness", "renewables_share_pct",
        "co2_per_capita_t", "gini_disposable", "carbon_price_usd",
        "rd_expenditure_gdp_pct", "democracy_index"
    ]
    coverage = panel.groupby("country_iso3")[key_cols].apply(lambda x: x.notna().mean())
    coverage = coverage.sort_index()

    fig, ax = plt.subplots(figsize=(12, max(8, len(coverage) * 0.22)))
    sns.heatmap(coverage, annot=False, cmap="YlGn", vmin=0, vmax=1,
                linewidths=0.3, ax=ax, cbar_kws={"label": "Coverage rate"})
    ax.set_title("Data Coverage by Country and Indicator", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results, save_path):
    """Bar chart comparing model performance across targets."""
    rows = []
    for target, info in results.items():
        label = info.get("label", target)
        for model_name, metrics in info["metrics"].items():
            rows.append({
                "Target": label[:30],
                "Model": model_name,
                "Val MAE": metrics["val_mae"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    targets = df["Target"].unique()
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, 3 * len(targets)), sharex=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = df[df["Target"] == target].sort_values("Val MAE")
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(subset))]
        ax.barh(subset["Model"], subset["Val MAE"], color=colors)
        ax.set_title(target, fontsize=11)
        ax.set_xlabel("Validation MAE")

    plt.suptitle("Model Performance Comparison (Validation MAE)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances, save_path, top_n=15):
    """Top feature importances for best models."""
    n_targets = len(importances)
    if n_targets == 0:
        return
    fig, axes = plt.subplots(1, min(n_targets, 4), figsize=(5 * min(n_targets, 4), 6))
    if n_targets == 1:
        axes = [axes]

    for ax, (target, imp) in zip(axes, list(importances.items())[:4]):
        if not imp:
            continue
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:top_n]
        names = [x[0][:25] for x in sorted_imp]
        vals = [x[1] for x in sorted_imp]
        ax.barh(names[::-1], vals[::-1], color=PALETTE[0])
        short_target = target.replace("target_", "").replace("_delta1y", " Δ1y").replace("_delta3y", " Δ3y")
        ax.set_title(short_target[:30], fontsize=10)
        ax.set_xlabel("Importance")

    plt.suptitle("Top Feature Importances (Gradient Boosting)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_case_study_roadmap(recs, country_iso3, save_path):
    """Gantt-style chart of intervention roadmap for a country."""
    roadmap = recs.get(country_iso3, {}).get("roadmap", [])
    if not roadmap:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(roadmap) * 0.6)))

    family_colors = {
        "infrastructure": PALETTE[0], "renewables": PALETTE[1],
        "fossil_exit": PALETTE[4], "efficiency": PALETTE[3],
        "electrification": PALETTE[5], "policy": PALETTE[2],
        "transport": PALETTE[6], "emissions": "#888888",
        "innovation": "#BB77AA", "storage": "#55AACC",
    }

    for i, item in enumerate(roadmap):
        tech_id = item["tech_id"]
        intv = next((iv for iv in __import__("src.policy.planner", fromlist=["INTERVENTIONS"]).INTERVENTIONS
                      if iv["tech_id"] == tech_id), None)
        if intv is None:
            continue
        lead = intv["lead_time_years"]
        lifetime = intv["lifetime_years"]
        color = family_colors.get(intv["family"], "#999999")

        ax.barh(i, lead, left=0, height=0.4, color=color, alpha=0.5, label="_nolegend_")
        ax.barh(i, min(lifetime, 20), left=lead, height=0.4, color=color, alpha=0.9)
        ax.text(lead + 0.3, i, f"U={item['utility']:.2f}", va="center", fontsize=8)

    ax.set_yticks(range(len(roadmap)))
    ax.set_yticklabels([r["name"] for r in roadmap], fontsize=9)
    ax.set_xlabel("Years from adoption decision")
    ax.set_title(f"Intervention Roadmap — {country_iso3}", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Legend
    handles = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items() if f in [r.get("family", "") for r in roadmap for iv in [next((x for x in __import__("src.policy.planner", fromlist=["INTERVENTIONS"]).INTERVENTIONS if x["tech_id"] == r["tech_id"]), {})]]]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_case_study_roadmap_simple(roadmap_items, country_iso3, interventions_list, save_path):
    """Simplified roadmap chart."""
    if not roadmap_items:
        return

    family_colors = {
        "infrastructure": PALETTE[0], "renewables": PALETTE[1],
        "fossil_exit": PALETTE[4], "efficiency": PALETTE[3],
        "electrification": PALETTE[5], "policy": PALETTE[2],
        "transport": PALETTE[6], "emissions": "#888888",
        "innovation": "#BB77AA", "storage": "#55AACC",
    }

    intv_map = {iv["tech_id"]: iv for iv in interventions_list}

    fig, ax = plt.subplots(figsize=(12, max(4, len(roadmap_items) * 0.55)))

    for i, item in enumerate(roadmap_items):
        intv = intv_map.get(item["tech_id"], {})
        lead = intv.get("lead_time_years", 2)
        lifetime = min(intv.get("lifetime_years", 15), 20)
        family = intv.get("family", "other")
        color = family_colors.get(family, "#999999")

        # Lead time (lighter)
        ax.barh(i, lead, left=0, height=0.45, color=color, alpha=0.35)
        # Active lifetime
        ax.barh(i, lifetime, left=lead, height=0.45, color=color, alpha=0.85)
        ax.text(lead + lifetime + 0.3, i, f"U={item['utility']:.2f}", va="center", fontsize=8, color="#333")

    ax.set_yticks(range(len(roadmap_items)))
    ax.set_yticklabels([r["name"] for r in roadmap_items], fontsize=9)
    ax.set_xlabel("Years from adoption decision")
    ax.set_title(f"Intervention Roadmap — {country_iso3}", fontsize=14, fontweight="bold")

    # Add legend
    seen = set()
    handles = []
    for item in roadmap_items:
        intv = intv_map.get(item["tech_id"], {})
        fam = intv.get("family", "other")
        if fam not in seen:
            handles.append(mpatches.Patch(color=family_colors.get(fam, "#999"), label=fam))
            seen.add(fam)
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_country_trajectories(panel, countries, indicator, save_path):
    """Line plot of an indicator over time for selected countries."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, iso3 in enumerate(countries):
        cdata = panel[panel["country_iso3"] == iso3].sort_values("year")
        ax.plot(cdata["year"], cdata[indicator], label=iso3,
                color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel(indicator.replace("_", " ").title())
    ax.set_title(f"{indicator.replace('_', ' ').title()} Over Time", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_uncertainty_fan(predictions, country_iso3, target_label, save_path):
    """Fan chart showing prediction with uncertainty bands."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(predictions["median"]))

    ax.fill_between(x, predictions["q10"], predictions["q90"],
                    alpha=0.2, color=PALETTE[0], label="80% interval")
    ax.fill_between(x, predictions["q25"], predictions["q75"],
                    alpha=0.3, color=PALETTE[0], label="50% interval")
    ax.plot(x, predictions["median"], color=PALETTE[0], linewidth=2, label="Median")

    ax.set_xlabel("Observation")
    ax.set_ylabel(target_label)
    ax.set_title(f"Prediction Uncertainty — {country_iso3}: {target_label}", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_figures(panel, results, importances, recs, interventions, fig_dir,
                        backtest_results=None):
    """Generate all report figures."""
    print("[VIZ] Generating figures...")

    plot_data_coverage(panel, f"{fig_dir}/data_coverage.png")
    print("  - data_coverage.png")

    if results:
        plot_model_comparison(results, f"{fig_dir}/model_comparison.png")
        print("  - model_comparison.png")

    if importances:
        plot_feature_importance(importances, f"{fig_dir}/feature_importance.png")
        print("  - feature_importance.png")

    # Country trajectories
    case_countries = ["POL", "DEU", "IND"]
    for indicator in ["renewables_share_pct", "co2_per_capita_t", "energy_intensity_mj_usd"]:
        fname = f"trajectory_{indicator}.png"
        plot_country_trajectories(panel, case_countries, indicator, f"{fig_dir}/{fname}")
        print(f"  - {fname}")

    # Roadmap charts
    intv_list = interventions if isinstance(interventions, list) else []
    for iso3 in case_countries:
        roadmap = recs.get(iso3, {}).get("roadmap", [])
        if roadmap:
            plot_case_study_roadmap_simple(roadmap, iso3, intv_list, f"{fig_dir}/roadmap_{iso3}.png")
            print(f"  - roadmap_{iso3}.png")

    # Backtesting figures
    if backtest_results:
        _plot_backtest_figures(backtest_results, fig_dir)

    # Ablation study figure
    _plot_ablation_study(fig_dir)

    # SHAP importance figure
    _plot_shap_from_saved(fig_dir)

    print(f"[VIZ] All figures saved to {fig_dir}/")


def _plot_ablation_study(fig_dir):
    """Generate ablation study figure from saved training results."""
    import os
    results_path = os.path.join(os.path.dirname(fig_dir), "..", "models_saved", "training_results.json")
    if not os.path.exists(results_path):
        return
    try:
        with open(results_path) as f:
            data = json.load(f)
        ablation = data.get("ablation", {})
        if not ablation:
            return

        rows = []
        for target, info in ablation.items():
            label = info.get("label", target)[:25]
            for key in ["no_governance_lift_pct", "no_culture_lift_pct", "no_gov_no_culture_lift_pct"]:
                if key in info:
                    group = key.replace("_lift_pct", "").replace("no_", "").replace("_", " + ").title()
                    rows.append({"Target": label, "Removed": group, "MAE Increase (%)": info[key]})

        if not rows:
            return

        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.3)))

        targets = df["Target"].unique()
        groups = df["Removed"].unique()
        x = np.arange(len(targets))
        width = 0.25

        for i, group in enumerate(groups):
            sub = df[df["Removed"] == group]
            vals = [sub[sub["Target"] == t]["MAE Increase (%)"].values[0]
                    if len(sub[sub["Target"] == t]) > 0 else 0 for t in targets]
            colors = [PALETTE[0] if v > 0 else PALETTE[4] for v in vals]
            offset = (i - len(groups)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=f"Remove: {group}",
                   color=PALETTE[i % len(PALETTE)], alpha=0.8)

        ax.set_ylabel("MAE Increase (%)")
        ax.set_title("Ablation Study: Feature Group Contribution", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/ablation_study.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  - ablation_study.png")
    except Exception as e:
        print(f"  - ablation_study.png SKIPPED ({e})")


def _plot_shap_from_saved(fig_dir):
    """Generate SHAP importance figure from saved training results."""
    import os
    results_path = os.path.join(os.path.dirname(fig_dir), "..", "models_saved", "training_results.json")
    if not os.path.exists(results_path):
        return
    try:
        with open(results_path) as f:
            data = json.load(f)
        shap_data = data.get("shap_importances", {})
        if not shap_data:
            return

        n_targets = min(len(shap_data), 3)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 6))
        if n_targets == 1:
            axes = [axes]

        for ax, (target, importances) in zip(axes, list(shap_data.items())[:n_targets]):
            sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:15]
            names = [x[0][:25] for x in sorted_imp]
            vals = [x[1] for x in sorted_imp]
            ax.barh(names[::-1], vals[::-1], color=PALETTE[0])
            short = target.replace("target_", "").replace("_delta1y", " Δ1y")[:25]
            ax.set_title(f"SHAP: {short}", fontsize=10)
            ax.set_xlabel("Mean |SHAP|")

        plt.suptitle("SHAP Feature Attribution", fontsize=13, y=1.02, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  - shap_importance.png")
    except Exception as e:
        print(f"  - shap_importance.png SKIPPED ({e})")


def _plot_backtest_figures(backtest_results, fig_dir):
    """Generate backtesting-specific figures."""

    # 1. Policy alignment heatmap
    alignment = backtest_results.get("policy_alignment", {})
    if alignment:
        _plot_alignment_summary(alignment, f"{fig_dir}/backtest_alignment.png")
        print("  - backtest_alignment.png")

    # 2. Forecast accuracy scatter
    forecast = backtest_results.get("forecast_accuracy", {})
    if forecast:
        _plot_forecast_accuracy(forecast, f"{fig_dir}/backtest_forecast_accuracy.png")
        print("  - backtest_forecast_accuracy.png")

    # 3. Stability chart
    stability = backtest_results.get("stability", {})
    if stability:
        _plot_stability(stability, f"{fig_dir}/backtest_stability.png")
        print("  - backtest_stability.png")


def _plot_alignment_summary(alignment, save_path):
    """Bar chart of alignment scores by country and horizon."""
    rows = []
    for iso3, data in alignment.items():
        for hz, hz_data in data.get("horizons", {}).items():
            wa = hz_data.get("weighted_alignment")
            ma = hz_data.get("mean_alignment")
            n = hz_data.get("n_interventions_evaluated", 0)
            if wa is not None:
                rows.append({"Country": iso3, "Horizon": hz, "Weighted Alignment": wa,
                             "Mean Alignment": ma, "N": n})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    countries = df["Country"].unique()
    horizons = df["Horizon"].unique()
    x = np.arange(len(countries))
    width = 0.35

    for i, hz in enumerate(horizons):
        sub = df[df["Horizon"] == hz]
        vals = [sub[sub["Country"] == c]["Weighted Alignment"].values[0]
                if len(sub[sub["Country"] == c]) > 0 else 0 for c in countries]
        offset = (i - len(horizons)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"Horizon: {hz}", color=PALETTE[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Weighted Alignment Score")
    ax.set_title("Retrospective Policy Alignment", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forecast_accuracy(forecast_summary, save_path):
    """Bar chart of forecast directional accuracy by target."""
    if not forecast_summary:
        return

    rows = []
    for key, stats in forecast_summary.items():
        rows.append({
            "Target": key.replace("_", " ").replace("delta", "Δ").title()[:30],
            "Direction Accuracy": stats.get("direction_accuracy", 0),
            "MAE": stats.get("mean_absolute_error", 0),
            "N": stats.get("n_countries", 0),
        })

    df = pd.DataFrame(rows).sort_values("Direction Accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.5)))

    colors = [PALETTE[0] if v >= 0.5 else PALETTE[4] for v in df["Direction Accuracy"]]
    ax.barh(df["Target"], df["Direction Accuracy"], color=colors)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Random baseline (50%)")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["Direction Accuracy"] + 0.02, i,
                f"{row['Direction Accuracy']:.0%} (n={row['N']})",
                va="center", fontsize=9)

    ax.set_xlabel("Directional Accuracy")
    ax.set_title("Forecast Backtest: Direction Accuracy by Target", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_stability(stability, save_path):
    """Bar chart of recommendation stability (Jaccard similarity)."""
    rows = []
    for iso3, data in stability.items():
        if "mean_jaccard" in data:
            rows.append({
                "Country": iso3,
                "Mean Jaccard": data["mean_jaccard"],
                "Top-3 Stable": data.get("top3_stable", False),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [PALETTE[0] if row["Top-3 Stable"] else PALETTE[2] for _, row in df.iterrows()]
    bars = ax.bar(df["Country"], df["Mean Jaccard"], color=colors)

    for bar, (_, row) in zip(bars, df.iterrows()):
        label = f"{row['Mean Jaccard']:.2f}"
        if row["Top-3 Stable"]:
            label += " ✓"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                label, ha="center", fontsize=10)

    ax.set_ylabel("Jaccard Similarity")
    ax.set_title("Recommendation Stability Across Base Years", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[0], label="Top-3 stable"),
                       Patch(facecolor=PALETTE[2], label="Top-3 changed")]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
