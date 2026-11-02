#!/usr/bin/env python3
"""
Main pipeline: data -> normalize -> features -> models -> planner -> report

Usage:
    python run_pipeline.py                  # default: real data
    python run_pipeline.py --real           # fetch from APIs
    python run_pipeline.py --synthetic      # use synthetic data (demo/testing)
    python run_pipeline.py --from-cache     # skip ingestion, use data_raw/ as-is
"""

import sys
import os
import json
import time
import argparse
import logging

# -- Project root --
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.features.engineer import engineer_features
from src.models.train import run_training
from src.policy.planner import run_recommendations, INTERVENTIONS
from src.policy.backtest import run_backtesting
from src.policy.nsga2 import run_nsga2_recommendations
from src.viz.plots import generate_all_figures
from src.report.generate_report import generate_report
from src.clean.normalize import normalize_panel

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "pipeline.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="SDG Skill-Trees Pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", default=True,
                      help="Ingest from real API endpoints (default)")
    mode.add_argument("--synthetic", action="store_true",
                      help="Use synthetic data for testing")
    mode.add_argument("--from-cache", action="store_true",
                      help="Skip ingestion, use existing data_raw/master_panel_raw.csv")
    p.add_argument("--skip-models", action="store_true",
                   help="Skip model training (use saved models)")
    p.add_argument("--countries", nargs="+", default=["POL", "DEU", "IND"],
                   help="Case study countries (ISO3)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    dirs = {
        "raw": os.path.join(ROOT, "data_raw"),
        "interim": os.path.join(ROOT, "data_interim"),
        "processed": os.path.join(ROOT, "data_processed"),
        "models": os.path.join(ROOT, "models_saved"),
        "recs": os.path.join(ROOT, "outputs", "recommendations"),
        "figures": os.path.join(ROOT, "reports", "figures"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  THESIS PIPELINE: SDG Skill-Trees for Decarbonization")
    logger.info("=" * 70)

    # -- Stage 1: Ingest --
    logger.info("")
    logger.info(">>> STAGE 1: Data Ingestion")

    if args.synthetic:
        logger.info("Mode: SYNTHETIC data")
        from src.ingest.synthetic_data import run_ingestion
        panel_raw, dim_country = run_ingestion(dirs["raw"])

    elif args.from_cache:
        logger.info("Mode: FROM CACHE (data_raw/)")
        import pandas as pd
        raw_path = os.path.join(dirs["raw"], "master_panel_raw.csv")
        if not os.path.exists(raw_path):
            logger.error(f"Cache file not found: {raw_path}")
            logger.error("Run with --real first to download data, or --synthetic for testing.")
            sys.exit(1)
        panel_raw = pd.read_csv(raw_path)
        dim_path = os.path.join(dirs["raw"], "dim_country.csv")
        dim_country = pd.read_csv(dim_path) if os.path.exists(dim_path) else pd.DataFrame()
        logger.info(f"Loaded {len(panel_raw)} rows from cache")

    else:
        logger.info("Mode: REAL data (API endpoints)")
        from src.ingest.real_data import run_ingestion_real
        panel_raw, dim_country = run_ingestion_real(dirs["raw"])
        # Save raw for caching
        panel_raw.to_csv(os.path.join(dirs["raw"], "master_panel_raw.csv"), index=False)
        dim_country.to_csv(os.path.join(dirs["raw"], "dim_country.csv"), index=False)

    # -- Stage 2: Normalize --
    logger.info("")
    logger.info(">>> STAGE 2: Normalization & Quality Checks")
    panel_clean = normalize_panel(panel_raw, output_dir=dirs["interim"])

    # -- Stage 3: Feature Engineering --
    logger.info("")
    logger.info(">>> STAGE 3: Feature Engineering")
    featured = engineer_features(panel_clean)
    featured.to_csv(os.path.join(dirs["processed"], "master_panel.csv"), index=False)
    dim_country.to_csv(os.path.join(dirs["processed"], "dim_country.csv"), index=False)
    logger.info(f"Saved master_panel.csv ({len(featured)} rows, {len(featured.columns)} cols)")

    # -- Stage 4: Model Training --
    logger.info("")
    logger.info(">>> STAGE 4: Model Training")
    if args.skip_models:
        logger.info("Skipped (--skip-models)")
        results, test_metrics, importances = {}, {}, {}
        results_path = os.path.join(dirs["models"], "training_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                saved = json.load(f)
                results = saved.get("results", {})
                test_metrics = saved.get("test_metrics", {})
        imp_path = os.path.join(dirs["models"], "feature_importances.json")
        if os.path.exists(imp_path):
            with open(imp_path) as f:
                importances = json.load(f)
    else:
        (all_models, results, test_metrics, importances,
         quantile_models, conformal_predictors, feature_cols) = run_training(
            featured, dirs["models"]
        )

    # -- Stage 5: Recommendations --
    logger.info("")
    logger.info(">>> STAGE 5: Intervention Recommendations")
    recs = run_recommendations(
        featured, dirs["recs"],
        case_study_countries=tuple(args.countries)
    )

    # -- Stage 5a: NSGA-II Multi-Objective Optimization --
    logger.info("")
    logger.info(">>> STAGE 5a: NSGA-II Pareto Optimization")
    try:
        pareto_results = run_nsga2_recommendations(
            featured, INTERVENTIONS, dirs["recs"],
            case_study_countries=tuple(args.countries)
        )
    except Exception as e:
        logger.warning(f"NSGA-II failed (non-fatal): {e}")
        pareto_results = {}

    # -- Stage 5b: Backtesting --
    logger.info("")
    logger.info(">>> STAGE 5b: Backtesting & Retrospective Analysis")
    backtest_results = {}
    if not args.skip_models:
        try:
            backtest_results = run_backtesting(
                featured_df=featured,
                panel_raw=panel_clean,
                models=all_models,
                feature_cols=feature_cols,
                recommendations=recs,
                output_dir=dirs["recs"],
                case_study_countries=tuple(args.countries),
                base_year=2017,
            )
        except Exception as e:
            logger.warning(f"Backtesting failed (non-fatal): {e}")
    else:
        logger.info("Skipped (--skip-models — no models available for backtest)")
        # Try to load cached results
        bt_path = os.path.join(dirs["recs"], "backtest_results.json")
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                backtest_results = json.load(f)

    # -- Stage 6: Visualization --
    logger.info("")
    logger.info(">>> STAGE 6: Visualization")
    generate_all_figures(
        panel_clean, results, importances, recs, INTERVENTIONS, dirs["figures"],
        backtest_results=backtest_results
    )

    # -- Stage 7: Report --
    logger.info("")
    logger.info(">>> STAGE 7: Report Generation")
    with open(os.path.join(ROOT, "registry", "datasets.json")) as f:
        datasets = json.load(f)["datasets"]

    report_path = os.path.join(ROOT, "reports", "final_report.html")
    generate_report(
        featured, results, test_metrics, importances, recs,
        INTERVENTIONS, datasets, dirs["figures"], report_path
    )

    # -- Done --
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("  Artifacts:")
    logger.info("    data_processed/master_panel.csv")
    logger.info("    models_saved/  (models, metrics, importances)")
    logger.info("    outputs/recommendations/  (roadmaps + backtest results)")
    logger.info("    reports/final_report.html")
    logger.info("    registry/datasets.json + citations.bib")
    logger.info("    pipeline.log")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
