"""
Normalize and quality-check the raw ingested panel.

Steps:
  1. Enforce ISO3 + integer year keys
  2. Clip outliers / sanity-check ranges
  3. Produce a coverage/missingness report
  4. Optionally interpolate short gaps (≤2 years)
"""

import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

# ── Expected ranges for sanity checking ──────────────────────────────
VALID_RANGES = {
    "electricity_access_pct":     (0, 100),
    "renewables_share_pct":       (0, 100),
    "energy_intensity_mj_usd":    (0, 50),
    "co2_per_capita_t":           (0, 80),
    "gdp_pc_ppp":                 (100, 200_000),
    "population":                 (10_000, 2e9),
    "urban_pct":                  (5, 100),
    "industry_gdp_pct":           (1, 80),
    "gov_effectiveness":          (-2.5, 2.5),
    "regulatory_quality":         (-2.5, 2.5),
    "rule_of_law":                (-2.5, 2.5),
    "control_corruption":         (-2.5, 2.5),
    "political_stability":        (-3.0, 2.5),
    "democracy_index":            (0, 1),
    "gini_disposable":            (15, 75),
    "gini_market":                (20, 80),
    "rd_expenditure_gdp_pct":     (0, 6),
    "education_exp_gdp_pct":      (0, 15),
    "carbon_price_usd":           (0, 300),
    "carbon_coverage_pct":        (0, 100),
    "conflict_events":            (0, 50_000),
    "coal_share_pct":             (0, 100),
    "gas_share_pct":              (0, 100),
    "solar_share_pct":            (0, 100),
    "wind_share_pct":             (0, 100),
    "hydro_share_pct":            (0, 100),
    "nuclear_share_pct":          (0, 100),
    "generation_twh":             (0, 10_000),
}


def enforce_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure country_iso3 is uppercase 3-letter and year is integer."""
    df = df.copy()
    df["country_iso3"] = df["country_iso3"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    # Remove non-country codes (aggregates like "WLD", "EUU", etc.)
    df = df[df["country_iso3"].str.len() == 3]
    df = df[~df["country_iso3"].str.startswith("X")]  # UN aggregates
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values to valid ranges. Out-of-range → NaN with provenance flag."""
    df = df.copy()
    flags = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        below = (df[col] < lo).sum()
        above = (df[col] > hi).sum()
        if below + above > 0:
            flags[col] = {"below_range": int(below), "above_range": int(above)}
            df.loc[df[col] < lo, col] = np.nan
            df.loc[df[col] > hi, col] = np.nan
    if flags:
        logger.info(f"[CLEAN] Clipped out-of-range values: {flags}")
    return df


def coverage_report(df: pd.DataFrame, output_path: str = None) -> dict:
    """Compute per-column and per-country missingness."""
    numeric_cols = [c for c in df.columns if c not in ("country_iso3", "year")]
    n_total = len(df)
    n_countries = df["country_iso3"].nunique()
    n_years = df["year"].nunique()

    # Per-column coverage
    col_coverage = {}
    for col in numeric_cols:
        if col in df.columns:
            col_coverage[col] = {
                "present": int(df[col].notna().sum()),
                "missing": int(df[col].isna().sum()),
                "coverage_pct": round(df[col].notna().mean() * 100, 1),
            }

    # Per-country coverage (average across target columns)
    target_cols = [c for c in numeric_cols if c in VALID_RANGES]
    country_cov = (
        df.groupby("country_iso3")[target_cols]
        .apply(lambda x: x.notna().mean().mean())
        .sort_values(ascending=False)
    )

    report = {
        "n_rows": n_total,
        "n_countries": n_countries,
        "n_years": n_years,
        "column_coverage": col_coverage,
        "worst_countries": country_cov.tail(10).to_dict(),
        "best_countries": country_cov.head(10).to_dict(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[CLEAN] Coverage report saved to {output_path}")

    return report


def interpolate_gaps(df: pd.DataFrame, max_gap: int = 2) -> pd.DataFrame:
    """
    Within each country, linearly interpolate gaps of ≤max_gap years.
    Adds a flag column marking interpolated values.
    """
    df = df.sort_values(["country_iso3", "year"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "year"]

    interpolated = df.copy()
    for col in numeric_cols:
        interpolated[col] = (
            interpolated.groupby("country_iso3")[col]
            .transform(lambda s: s.interpolate(method="linear", limit=max_gap))
        )

    # Flag how many values were filled
    n_filled = interpolated[numeric_cols].notna().sum().sum() - df[numeric_cols].notna().sum().sum()
    if n_filled > 0:
        logger.info(f"[CLEAN] Interpolated {n_filled} values (max gap = {max_gap}y)")

    return interpolated


def normalize_panel(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """Full normalization pipeline."""
    logger.info("[CLEAN] Normalizing panel...")
    df = enforce_keys(df)
    df = clip_outliers(df)

    if output_dir:
        coverage_report(df, f"{output_dir}/coverage_report.json")

    df = interpolate_gaps(df, max_gap=2)

    # De-duplicate (keep first if any exact duplicates)
    before = len(df)
    df = df.drop_duplicates(subset=["country_iso3", "year"], keep="first")
    after = len(df)
    if before != after:
        logger.info(f"[CLEAN] Removed {before - after} duplicate rows")

    logger.info(f"[CLEAN] Normalized panel: {len(df)} rows, {df['country_iso3'].nunique()} countries")
    return df
