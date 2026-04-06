#!/usr/bin/env python3
"""
Post-run verification: validate that a pipeline run produced correct outputs.

Usage:
    python verify_run.py                  # check most recent run
    python verify_run.py --strict         # fail on any warning
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

def _ok(msg):    print(f"  \033[32m✓\033[0m {msg}")
def _warn(msg):  print(f"  \033[33m⚠\033[0m {msg}")
def _fail(msg):  print(f"  \033[31m✗\033[0m {msg}")


def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    print("\n\033[1m  Post-run verification\033[0m\n")
    issues = []

    # 1. Check output files exist
    required_files = {
        "data_processed/master_panel.csv": 100_000,   # min bytes
        "models_saved/training_results.json": 500,
        "models_saved/feature_importances.json": 100,
        "models_saved/models.pkl": 1000,
        "outputs/recommendations/all_recommendations.json": 500,
        "reports/final_report.html": 10_000,
        "registry/datasets.json": 500,
        "registry/citations.bib": 500,
    }

    for path, min_size in required_files.items():
        full = ROOT / path
        if full.exists():
            size = full.stat().st_size
            if size >= min_size:
                _ok(f"{path} ({size:,} bytes)")
            else:
                _warn(f"{path} exists but seems small ({size:,} bytes, expected ≥{min_size:,})")
                issues.append(f"small: {path}")
        else:
            _fail(f"{path} MISSING")
            issues.append(f"missing: {path}")

    # 2. Check case study roadmaps
    rec_dir = ROOT / "outputs" / "recommendations"
    for iso3 in ["POL", "DEU", "IND"]:
        csv_path = rec_dir / f"{iso3}_roadmap.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            if len(df) >= 5 and "utility" in df.columns:
                _ok(f"{iso3} roadmap: {len(df)} interventions")
            else:
                _warn(f"{iso3} roadmap: only {len(df)} rows or missing 'utility' column")
                issues.append(f"weak: {iso3} roadmap")
        else:
            _fail(f"{iso3}_roadmap.csv MISSING")
            issues.append(f"missing: {iso3}_roadmap.csv")

    # 3. Check training results quality
    results_path = ROOT / "models_saved" / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        results = data.get("results", {})
        test_metrics = data.get("test_metrics", {})

        n_targets = len(results)
        if n_targets >= 5:
            _ok(f"Trained {n_targets} targets")
        else:
            _warn(f"Only {n_targets} targets trained (expected ≥5)")
            issues.append("few targets")

        # Check that GBR beats naive on at least some targets
        gbr_wins = 0
        for target, info in results.items():
            metrics = info.get("metrics", {})
            naive_mae = metrics.get("naive_persistence", {}).get("val_mae", float("inf"))
            best = info.get("best_model", "")
            best_mae = metrics.get(best, {}).get("val_mae", float("inf"))
            if best_mae < naive_mae:
                gbr_wins += 1

        if gbr_wins >= n_targets * 0.5:
            _ok(f"Models beat naive baseline on {gbr_wins}/{n_targets} targets")
        else:
            _warn(f"Models beat naive on only {gbr_wins}/{n_targets} targets")
            issues.append("weak model performance")

        # Check conformal coverage
        for target, tm in test_metrics.items():
            cov = tm.get("conformal_coverage_90pct")
            if cov is not None and cov < 0.70:
                _warn(f"Low conformal coverage for {target}: {cov:.0%}")
                issues.append(f"low coverage: {target}")

    # 4. Check coverage report
    coverage_path = ROOT / "data_interim" / "coverage_report.json"
    if coverage_path.exists():
        with open(coverage_path) as f:
            cov = json.load(f)
        n_countries = cov.get("n_countries", 0)
        if n_countries >= 100:
            _ok(f"Panel covers {n_countries} countries")
        elif n_countries >= 40:
            _warn(f"Panel covers only {n_countries} countries (expected ≥100 with real data)")
            issues.append("few countries")
        else:
            _fail(f"Panel covers only {n_countries} countries — merge likely failed")
            issues.append("very few countries")

        # Check key column coverage
        col_cov = cov.get("column_coverage", {})
        for key_col in ["gdp_pc_ppp", "gov_effectiveness", "co2_per_capita_t"]:
            pct = col_cov.get(key_col, {}).get("coverage_pct", 0)
            if pct >= 50:
                _ok(f"{key_col} coverage: {pct}%")
            else:
                _warn(f"{key_col} coverage: {pct}% (low)")
                issues.append(f"low coverage: {key_col}")
    else:
        _warn("No coverage report found (data_interim/coverage_report.json)")

    # 5. Check figures
    fig_dir = ROOT / "reports" / "figures"
    expected_figs = [
        "data_coverage.png", "model_comparison.png", "feature_importance.png",
        "roadmap_POL.png", "roadmap_DEU.png", "roadmap_IND.png",
    ]
    for fig in expected_figs:
        if (fig_dir / fig).exists():
            _ok(f"Figure: {fig}")
        else:
            _warn(f"Figure missing: {fig}")
            issues.append(f"missing fig: {fig}")

    # Summary
    print()
    if not issues:
        _ok("ALL CHECKS PASSED")
        return 0
    elif args.strict:
        _fail(f"STRICT MODE: {len(issues)} issues found")
        for i in issues:
            print(f"    - {i}")
        return 1
    else:
        _warn(f"{len(issues)} non-blocking issues (use --strict to fail on these)")
        for i in issues:
            print(f"    - {i}")
        return 0


if __name__ == "__main__":
    sys.exit(verify())
