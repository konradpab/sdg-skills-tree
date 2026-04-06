"""
Feature engineering for the master panel.
Produces: lags, deltas, rolling means, composite indices, target variables.
"""

import numpy as np
import pandas as pd


# ─── Target indicator columns ───────────────────────────────────────
TARGET_COLS = [
    "renewables_share_pct",
    "co2_per_capita_t",
    "energy_intensity_mj_usd",
    "electricity_access_pct",
    "gini_disposable",
]

# ─── Feature columns (raw) ──────────────────────────────────────────
CAPABILITY_COLS = [
    "gdp_pc_ppp", "gov_effectiveness", "regulatory_quality", "rule_of_law",
    "control_corruption", "democracy_index", "rd_expenditure_gdp_pct",
    "education_exp_gdp_pct", "urban_pct",
]
STRUCTURE_COLS = [
    "industry_gdp_pct", "manufacturing_va_gdp_pct", "population",
    "coal_share_pct", "gas_share_pct", "generation_twh",
]
POLICY_COLS = [
    "carbon_price_usd", "carbon_coverage_pct",
]
CONSTRAINT_COLS = [
    "conflict_events", "political_stability",
]

# ─── Cultural dimensions (optional — merged when available) ──────
CULTURE_COLS = [
    "hofstede_pdi", "hofstede_idv", "hofstede_mas", "hofstede_uai",
    "hofstede_lto", "hofstede_ivr",
    "minkov_idv_coll", "minkov_flx_mon",
    "culture_emancipation", "culture_pragmatism", "culture_inst_trust",
]


def add_lags(df, cols, lags=(1, 2)):
    """Add lagged features per country."""
    df = df.sort_values(["country_iso3", "year"])
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("country_iso3")[col].shift(lag)
    return df


def add_deltas(df, cols, horizons=(1, 3)):
    """Add backward-looking deltas (change from t-h to t)."""
    df = df.sort_values(["country_iso3", "year"])
    for col in cols:
        for h in horizons:
            lagged = df.groupby("country_iso3")[col].shift(h)
            df[f"{col}_delta{h}y"] = df[col] - lagged
    return df


def add_rolling(df, cols, window=3):
    """Add rolling mean features."""
    df = df.sort_values(["country_iso3", "year"])
    for col in cols:
        df[f"{col}_roll{window}"] = (
            df.groupby("country_iso3")[col]
            .transform(lambda x: x.rolling(window, min_periods=2).mean())
        )
    return df


def add_forward_targets(df, cols, horizons=(1, 3)):
    """Add forward-looking target deltas for forecasting."""
    df = df.sort_values(["country_iso3", "year"])
    for col in cols:
        for h in horizons:
            future = df.groupby("country_iso3")[col].shift(-h)
            df[f"target_{col}_delta{h}y"] = future - df[col]
    return df


def compute_capability_index(df):
    """Composite capability score (0-1) from governance + GDP + education."""
    gov_cols = ["gov_effectiveness", "regulatory_quality", "rule_of_law", "control_corruption"]
    gov_mean = df[gov_cols].mean(axis=1)
    # Normalize GDP per capita to 0-1 (log scale)
    gdp_norm = np.log1p(df["gdp_pc_ppp"]) / np.log1p(80000)
    edu_norm = df["education_exp_gdp_pct"].clip(0, 8) / 8
    rd_norm = df["rd_expenditure_gdp_pct"].clip(0, 5) / 5
    # Weighted composite
    df["capability_index"] = (
        0.35 * ((gov_mean + 0.5) / 1.5).clip(0, 1)  # rescale from [-0.5,1] to [0,1]
        + 0.25 * gdp_norm.clip(0, 1)
        + 0.20 * edu_norm.clip(0, 1)
        + 0.20 * rd_norm.clip(0, 1)
    )
    return df


def compute_feasibility_index(df):
    """Feasibility score combining capability, stability, and conflict."""
    stability_norm = ((df["political_stability"] + 1) / 2).clip(0, 1)
    conflict_penalty = 1 - (df["conflict_events"].clip(0, 200) / 200)
    df["feasibility_index"] = (
        0.50 * df["capability_index"]
        + 0.30 * stability_norm
        + 0.20 * conflict_penalty
    )
    return df


def compute_green_transition_index(df):
    """How far along the green transition a country is."""
    ren_norm = df["renewables_share_pct"].clip(0, 100) / 100
    coal_penalty = 1 - df["coal_share_pct"].clip(0, 100) / 100
    ei_norm = 1 - (df["energy_intensity_mj_usd"].clip(1, 20) - 1) / 19
    cp_norm = df["carbon_coverage_pct"].clip(0, 100) / 100
    df["green_transition_index"] = (
        0.30 * ren_norm + 0.25 * coal_penalty + 0.25 * ei_norm + 0.20 * cp_norm
    )
    return df


def engineer_features(panel):
    """Full feature engineering pipeline."""
    print("[FEATURES] Engineering features...")
    df = panel.copy()

    # ── Merge cultural dimensions (static, country-level) ─────────
    try:
        from src.ingest.cultural_values import merge_cultural_features, compute_cultural_composites
        df = merge_cultural_features(df)
        df = compute_cultural_composites(df)
        culture_present = any(c in df.columns for c in CULTURE_COLS)
        if culture_present:
            n_culture = sum(1 for c in CULTURE_COLS if c in df.columns)
            print(f"[FEATURES] {n_culture} cultural dimensions merged")
    except Exception as e:
        print(f"[FEATURES] Cultural features skipped: {e}")

    all_numeric = [c for c in df.columns if c not in ("country_iso3", "year") and df[c].dtype in ("float64", "int64")]

    # Lags & deltas for key features
    key_cols = CAPABILITY_COLS + STRUCTURE_COLS + POLICY_COLS + TARGET_COLS
    key_cols = [c for c in key_cols if c in df.columns]

    df = add_lags(df, key_cols, lags=(1, 2))
    df = add_deltas(df, key_cols, horizons=(1, 3))
    df = add_rolling(df, key_cols, window=3)

    # Forward targets
    df = add_forward_targets(df, TARGET_COLS, horizons=(1, 3, 5))

    # Composite indices
    df = compute_capability_index(df)
    df = compute_feasibility_index(df)
    df = compute_green_transition_index(df)

    # Log-transform skewed features
    for col in ["gdp_pc_ppp", "population", "generation_twh", "patent_applications", "conflict_events"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    n_features = len([c for c in df.columns if c not in ("country_iso3", "year")])
    print(f"[FEATURES] {n_features} features generated, {len(df)} rows")
    return df


def get_feature_names(df, target_prefix="target_"):
    """Return feature column names (excluding targets and identifiers)."""
    exclude = {"country_iso3", "year"}
    target_cols = [c for c in df.columns if c.startswith(target_prefix)]
    exclude.update(target_cols)
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "float32", "int64")]


if __name__ == "__main__":
    panel = pd.read_csv("/home/claude/thesis/data_raw/master_panel_raw.csv")
    featured = engineer_features(panel)
    featured.to_csv("/home/claude/thesis/data_processed/master_panel_featured.csv", index=False)
    print(f"[FEATURES] Saved to data_processed/master_panel_featured.csv")
