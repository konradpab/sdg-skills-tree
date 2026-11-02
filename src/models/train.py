"""
Model training with time-aware validation, baselines, gradient boosting,
stacking ensemble, quantile regression for uncertainty, and conformal prediction.
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
    StackingRegressor
)
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy import stats


# ─── Configuration ───────────────────────────────────────────────────
TARGET_NAMES = {
    "target_renewables_share_pct_delta1y": "Δ Renewables Share (1y)",
    "target_co2_per_capita_t_delta1y": "Δ CO₂ per capita (1y)",
    "target_energy_intensity_mj_usd_delta1y": "Δ Energy Intensity (1y)",
    "target_electricity_access_pct_delta1y": "Δ Electricity Access (1y)",
    "target_gini_disposable_delta1y": "Δ Gini Disposable (1y)",
    "target_renewables_share_pct_delta3y": "Δ Renewables Share (3y)",
    "target_co2_per_capita_t_delta3y": "Δ CO₂ per capita (3y)",
    "target_energy_intensity_mj_usd_delta3y": "Δ Energy Intensity (3y)",
    "target_electricity_access_pct_delta3y": "Δ Electricity Access (3y)",
    "target_gini_disposable_delta3y": "Δ Gini Disposable (3y)",
    "target_renewables_share_pct_delta5y": "Δ Renewables Share (5y)",
    "target_co2_per_capita_t_delta5y": "Δ CO₂ per capita (5y)",
    "target_energy_intensity_mj_usd_delta5y": "Δ Energy Intensity (5y)",
    "target_electricity_access_pct_delta5y": "Δ Electricity Access (5y)",
    "target_gini_disposable_delta5y": "Δ Gini Disposable (5y)",
}

TRAIN_CUTOFF = 2017
VAL_CUTOFF = 2020  # Train: <=2017, Val: 2018-2020, Test: 2021+


def get_time_splits(df):
    """Time-aware train/val/test split (no leakage)."""
    train = df[df["year"] <= TRAIN_CUTOFF].copy()
    val = df[(df["year"] > TRAIN_CUTOFF) & (df["year"] <= VAL_CUTOFF)].copy()
    test = df[df["year"] > VAL_CUTOFF].copy()
    return train, val, test


class NaivePersistence:
    """Baseline: predict Δ = 0 (no change)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))


class ModelSuite:
    """Train and evaluate multiple models for a single target."""

    def __init__(self, target_col, feature_cols):
        self.target = target_col
        self.features = feature_cols
        self.models = {}
        self.metrics = {}
        self.feature_importances = {}

    def _clean_xy(self, df):
        subset = df[self.features + [self.target]].dropna()
        return subset[self.features].values, subset[self.target].values, subset.index

    def train_all(self, train_df, val_df):
        """Train baselines + core models + stacking ensemble."""
        X_train, y_train, _ = self._clean_xy(train_df)
        X_val, y_val, _ = self._clean_xy(val_df)

        if len(X_train) < 30 or len(X_val) < 10:
            print(f"  [SKIP] {self.target}: insufficient data (train={len(X_train)}, val={len(X_val)})")
            return False

        model_specs = {
            "naive_persistence": NaivePersistence(),
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            "hist_gb": HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.05, random_state=42
            ),
        }

        for name, model in model_specs.items():
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            self.models[name] = model
            self.metrics[name] = {
                "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "val_mae": float(mean_absolute_error(y_val, y_pred_val)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                "val_rank_corr": float(stats.spearmanr(y_val, y_pred_val).statistic) if len(y_val) > 5 else np.nan,
            }

            # Feature importance for tree models
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(self.features, model.feature_importances_))
                self.feature_importances[name] = dict(sorted(imp.items(), key=lambda x: -x[1])[:20])

        # ── Stacking Ensemble ────────────────────────────────────────
        # Combine predictions from the 4 learnable models (not naive)
        # using Ridge regression as the meta-learner.
        try:
            estimators = [
                ("elastic_net", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
                ("random_forest", RandomForestRegressor(n_estimators=80, max_depth=7, random_state=42, n_jobs=-1)),
                ("gradient_boosting", GradientBoostingRegressor(
                    n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)),
                ("hist_gb", HistGradientBoostingRegressor(
                    max_iter=150, max_depth=6, learning_rate=0.05, random_state=42)),
            ]
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
                cv=3,                # 3-fold CV for meta-features (time-aware would be better but this is pragmatic)
                passthrough=False,   # only use base model predictions as meta-features
                n_jobs=-1,
            )
            stack.fit(X_train, y_train)
            y_pred_train_stack = stack.predict(X_train)
            y_pred_val_stack = stack.predict(X_val)

            self.models["ensemble_stack"] = stack
            self.metrics["ensemble_stack"] = {
                "train_mae": float(mean_absolute_error(y_train, y_pred_train_stack)),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train_stack))),
                "val_mae": float(mean_absolute_error(y_val, y_pred_val_stack)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, y_pred_val_stack))),
                "val_rank_corr": float(stats.spearmanr(y_val, y_pred_val_stack).statistic) if len(y_val) > 5 else np.nan,
            }
            print(f"  Ensemble stack val_mae={self.metrics['ensemble_stack']['val_mae']:.4f}")
        except Exception as e:
            print(f"  [WARN] Ensemble training failed: {e}")

        return True

    def get_best_model(self, metric="val_mae"):
        """Return name of best model by validation metric."""
        valid = {k: v[metric] for k, v in self.metrics.items() if not np.isnan(v.get(metric, np.nan))}
        if not valid:
            return "gradient_boosting"
        return min(valid, key=valid.get)


class QuantileModel:
    """Uncertainty estimation via quantile regression with gradient boosting."""

    def __init__(self, feature_cols, quantiles=(0.1, 0.5, 0.9)):
        self.features = feature_cols
        self.quantiles = quantiles
        self.models = {}

    def fit(self, train_df, target_col):
        subset = train_df[self.features + [target_col]].dropna()
        X = subset[self.features].values
        y = subset[target_col].values
        for q in self.quantiles:
            loss = "quantile"
            model = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                loss=loss, alpha=q, random_state=42
            )
            model.fit(X, y)
            self.models[q] = model

    def predict(self, X):
        """Return dict of quantile predictions."""
        return {q: model.predict(X) for q, model in self.models.items()}


class ConformalPredictor:
    """Split conformal prediction for calibrated intervals."""

    def __init__(self, base_model, alpha=0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.residuals = None

    def calibrate(self, X_cal, y_cal):
        """Compute nonconformity scores on calibration set."""
        preds = self.base_model.predict(X_cal)
        self.residuals = np.abs(y_cal - preds)

    def predict_interval(self, X):
        """Return (lower, point, upper) predictions."""
        preds = self.base_model.predict(X)
        if self.residuals is None:
            raise ValueError("Must calibrate first")
        q = np.quantile(self.residuals, 1 - self.alpha)
        return preds - q, preds, preds + q


def run_training(featured_df, save_dir):
    """Main training entry point."""
    print("[TRAIN] Starting model training...")

    feature_cols = [c for c in featured_df.columns
                    if c not in ("country_iso3", "year")
                    and not c.startswith("target_")
                    and featured_df[c].dtype in ("float64", "float32", "int64")]

    train_df, val_df, test_df = get_time_splits(featured_df)
    print(f"[TRAIN] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    all_results = {}
    all_models = {}
    all_importances = {}
    conformal_predictors = {}
    quantile_models = {}

    targets = [c for c in featured_df.columns if c.startswith("target_") and "delta1y" in c]
    targets += [c for c in featured_df.columns if c.startswith("target_") and "delta3y" in c]
    targets += [c for c in featured_df.columns if c.startswith("target_") and "delta5y" in c]

    for target in targets:
        label = TARGET_NAMES.get(target, target)
        print(f"\n  Training for: {label}")

        suite = ModelSuite(target, feature_cols)
        success = suite.train_all(train_df, val_df)
        if not success:
            continue

        best_name = suite.get_best_model()
        best_model = suite.models[best_name]
        print(f"  Best model: {best_name} (val_mae={suite.metrics[best_name]['val_mae']:.4f})")

        all_results[target] = {
            "label": label,
            "metrics": suite.metrics,
            "best_model": best_name,
        }
        all_models[target] = best_model
        all_importances[target] = suite.feature_importances.get(best_name, {})

        # Quantile model for uncertainty
        print(f"  Training quantile models...")
        qm = QuantileModel(feature_cols, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
        qm.fit(train_df, target)
        quantile_models[target] = qm

        # Conformal prediction
        X_val_clean = val_df[feature_cols + [target]].dropna()
        if len(X_val_clean) > 20:
            cp = ConformalPredictor(best_model, alpha=0.1)
            cp.calibrate(X_val_clean[feature_cols].values, X_val_clean[target].values)
            conformal_predictors[target] = cp
            print(f"  Conformal predictor calibrated (n_cal={len(X_val_clean)})")

    # ── Ablation study ─────────────────────────────────────────────
    # Test whether governance and cultural features add predictive lift
    print("\n[TRAIN] Running ablation study...")
    governance_cols = [c for c in feature_cols if any(g in c for g in
        ["gov_effectiveness", "regulatory_quality", "rule_of_law",
         "control_corruption", "political_stability", "democracy",
         "voice_accountability", "capability_index", "feasibility_index"])]
    culture_cols = [c for c in feature_cols if any(g in c for g in
        ["hofstede_", "minkov_", "culture_"])]

    ablation_results = {}
    # Pick 3 representative 1y targets for ablation
    ablation_targets = [t for t in targets if "delta1y" in t][:3]
    for target in ablation_targets:
        label = TARGET_NAMES.get(target, target)
        ablation_results[target] = {"label": label}

        for ablation_name, cols_to_remove in [
            ("no_governance", governance_cols),
            ("no_culture", culture_cols),
            ("no_gov_no_culture", governance_cols + culture_cols),
        ]:
            reduced_cols = [c for c in feature_cols if c not in cols_to_remove]
            if len(reduced_cols) < 10:
                continue
            try:
                sub_train = train_df[reduced_cols + [target]].dropna()
                sub_val = val_df[reduced_cols + [target]].dropna()
                if len(sub_train) < 30 or len(sub_val) < 10:
                    continue
                m = HistGradientBoostingRegressor(max_iter=150, max_depth=5, learning_rate=0.05, random_state=42)
                m.fit(sub_train[reduced_cols].values, sub_train[target].values)
                pred = m.predict(sub_val[reduced_cols].values)
                abl_mae = float(mean_absolute_error(sub_val[target].values, pred))
                ablation_results[target][ablation_name] = abl_mae

                # Compare with full model
                full_mae = all_results.get(target, {}).get("metrics", {}).get("hist_gb", {}).get("val_mae")
                if full_mae:
                    lift = (abl_mae - full_mae) / full_mae * 100
                    ablation_results[target][f"{ablation_name}_lift_pct"] = round(lift, 2)
                    print(f"  {label[:25]} | {ablation_name}: MAE={abl_mae:.4f} (Δ={lift:+.1f}%)")
            except Exception as e:
                print(f"  {label[:25]} | {ablation_name}: failed ({e})")

    # ── SHAP feature importance ─────────────────────────────────
    shap_results = {}
    try:
        import shap
        print("\n[TRAIN] Computing SHAP values...")
        for target in ablation_targets:
            model = all_models.get(target)
            if model is None:
                continue
            clean = val_df[feature_cols + [target]].dropna()
            if len(clean) < 20:
                continue
            X_shap = clean[feature_cols].values[:200]  # subsample for speed
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                # Mean absolute SHAP value per feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(-mean_abs_shap)[:20]
                shap_results[target] = {
                    feature_cols[i]: round(float(mean_abs_shap[i]), 6) for i in top_idx
                }
                print(f"  SHAP computed for {TARGET_NAMES.get(target, target)[:30]}")
            except Exception as e:
                print(f"  SHAP failed for {target}: {e}")
    except ImportError:
        print("[TRAIN] SHAP not installed — skipping (pip install shap)")

    # Test set evaluation
    print("\n[TRAIN] Test set evaluation...")
    test_metrics = {}
    for target, model in all_models.items():
        test_clean = test_df[feature_cols + [target]].dropna()
        if len(test_clean) < 5:
            continue
        X_test = test_clean[feature_cols].values
        y_test = test_clean[target].values
        y_pred = model.predict(X_test)
        test_metrics[target] = {
            "test_mae": float(mean_absolute_error(y_test, y_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "test_rank_corr": float(stats.spearmanr(y_test, y_pred).statistic) if len(y_test) > 5 else np.nan,
        }
        # Add conformal coverage
        if target in conformal_predictors:
            lo, _, hi = conformal_predictors[target].predict_interval(X_test)
            coverage = np.mean((y_test >= lo) & (y_test <= hi))
            test_metrics[target]["conformal_coverage_90pct"] = float(coverage)

    # Save everything
    with open(f"{save_dir}/training_results.json", "w") as f:
        json.dump({
            "results": all_results,
            "test_metrics": test_metrics,
            "ablation": ablation_results,
            "shap_importances": shap_results,
        }, f, indent=2, default=str)

    with open(f"{save_dir}/models.pkl", "wb") as f:
        pickle.dump(all_models, f)

    with open(f"{save_dir}/feature_importances.json", "w") as f:
        json.dump(all_importances, f, indent=2)

    with open(f"{save_dir}/quantile_models.pkl", "wb") as f:
        pickle.dump(quantile_models, f)

    with open(f"{save_dir}/conformal_predictors.pkl", "wb") as f:
        pickle.dump(conformal_predictors, f)

    print(f"\n[TRAIN] Complete. {len(all_models)} targets trained, results saved to {save_dir}/")
    return all_models, all_results, test_metrics, all_importances, quantile_models, conformal_predictors, feature_cols
