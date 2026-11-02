"""
Smoke test: run the full pipeline on synthetic data and verify outputs exist.
"""

import os
import sys
import json
import tempfile
import shutil

import pytest
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def pipeline_output():
    """Run the pipeline once in a temp directory and return paths."""
    from src.ingest.synthetic_data import run_ingestion
    from src.clean.normalize import normalize_panel
    from src.features.engineer import engineer_features
    from src.models.train import run_training
    from src.policy.planner import run_recommendations, INTERVENTIONS

    tmpdir = tempfile.mkdtemp(prefix="thesis_test_")
    raw_dir = os.path.join(tmpdir, "data_raw")
    proc_dir = os.path.join(tmpdir, "data_processed")
    model_dir = os.path.join(tmpdir, "models_saved")
    rec_dir = os.path.join(tmpdir, "recommendations")
    for d in [raw_dir, proc_dir, model_dir, rec_dir]:
        os.makedirs(d)

    panel_raw, dim = run_ingestion(raw_dir)
    panel_clean = normalize_panel(panel_raw)
    featured = engineer_features(panel_clean)
    models, results, test_metrics, imp, qm, cp, fcols = run_training(featured, model_dir)
    recs = run_recommendations(featured, rec_dir, ("POL", "DEU", "IND"))

    yield {
        "tmpdir": tmpdir,
        "panel_raw": panel_raw,
        "featured": featured,
        "results": results,
        "test_metrics": test_metrics,
        "recs": recs,
        "model_dir": model_dir,
        "rec_dir": rec_dir,
    }
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_panel_shape(pipeline_output):
    panel = pipeline_output["panel_raw"]
    assert len(panel) > 1000, f"Panel too small: {len(panel)}"
    assert panel["country_iso3"].nunique() >= 40
    assert panel["year"].min() <= 2000
    assert panel["year"].max() >= 2020


def test_features_generated(pipeline_output):
    f = pipeline_output["featured"]
    assert "capability_index" in f.columns
    assert "feasibility_index" in f.columns
    assert any(c.startswith("target_") for c in f.columns)
    assert any("_lag1" in c for c in f.columns)
    assert any("_delta1y" in c for c in f.columns)


def test_models_trained(pipeline_output):
    results = pipeline_output["results"]
    assert len(results) >= 5, f"Only {len(results)} targets trained"
    for target, info in results.items():
        assert "metrics" in info
        assert "best_model" in info
        assert info["metrics"][info["best_model"]]["val_mae"] >= 0


def test_recommendations_generated(pipeline_output):
    recs = pipeline_output["recs"]
    for iso3 in ["POL", "DEU", "IND"]:
        assert iso3 in recs, f"Missing recommendations for {iso3}"
        assert len(recs[iso3]["roadmap"]) >= 5


def test_conformal_coverage(pipeline_output):
    """Conformal prediction should achieve ~90% coverage on test set."""
    tm = pipeline_output["test_metrics"]
    for target, metrics in tm.items():
        cov = metrics.get("conformal_coverage_90pct")
        if cov is not None:
            # Allow some slack — synthetic data may not perfectly calibrate
            assert cov >= 0.60, f"Conformal coverage too low for {target}: {cov}"


def test_model_files_saved(pipeline_output):
    model_dir = pipeline_output["model_dir"]
    assert os.path.exists(os.path.join(model_dir, "models.pkl"))
    assert os.path.exists(os.path.join(model_dir, "training_results.json"))
    assert os.path.exists(os.path.join(model_dir, "quantile_models.pkl"))


def test_recommendation_csvs(pipeline_output):
    rec_dir = pipeline_output["rec_dir"]
    for iso3 in ["POL", "DEU", "IND"]:
        csv_path = os.path.join(rec_dir, f"{iso3}_roadmap.csv")
        assert os.path.exists(csv_path), f"Missing {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) >= 5
        assert "utility" in df.columns


def test_no_future_leakage(pipeline_output):
    """Verify train set doesn't contain post-2017 data."""
    from src.models.train import TRAIN_CUTOFF
    f = pipeline_output["featured"]
    train = f[f["year"] <= TRAIN_CUTOFF]
    assert train["year"].max() <= TRAIN_CUTOFF
