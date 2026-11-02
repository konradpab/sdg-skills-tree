"""
Cultural values data layer.

Provides Hofstede's 6 cultural dimensions plus two Minkov & Kaasa-style
meta-dimensions (Individualism-Collectivism ≈ Emancipation,
Flexibility-Monumentalism ≈ Long-Term Orientation), loaded from a
user-supplied CSV (data_raw/culture/hofstede_scores.csv). The scores are
not redistributed with this repository; see DATA_SOURCES.md.

Note (thesis v1.2 erratum): in the thesis runs, the two Minkov-Kaasa
columns were approximated from Hofstede IDV and LTO (correlation > 0.99),
not taken from the published Minkov & Kaasa (2022) scores. The culture
findings therefore rest mainly on the Hofstede dimensions. Supply the
published scores in the CSV for a faithful Minkov-Kaasa run.

Design decisions (addressing Hofstede critique):
  - Dimensions are treated as CONTEXT features, never as deterministic
    constraints or causal drivers.
  - We include Minkov & Kaasa's reclassification to address
    McSweeney (2002) and Williamson (2002) critiques of the original
    Hofstede model. Their factor analysis of WVS data produces two
    robust meta-dimensions that are more stable across waves.
  - Culture features enter the model as soft heterogeneity signals
    alongside governance, GDP, and structural variables. The ablation
    study tests whether they add predictive lift above those controls.
  - The code explicitly documents that these are country-level
    aggregates — they describe statistical tendencies in populations,
    not deterministic properties of individuals.

Sources:
  - Hofstede, G. (2001). Culture's Consequences (2nd ed.)
  - Hofstede Insights: https://www.hofstede-insights.com/
  - Minkov, M. & Kaasa, A. (2022). "A revision of Hofstede's model"
    Journal of International Management, 28(4).
  - Schwartz, S.H. (2012). "An overview of the Schwartz theory"
  - World Values Survey Wave 7 (2017-2022)
"""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════
#  Hofstede 6 dimensions + Minkov-Kaasa reclassification
#  Scores: 0-100 for Hofstede; standardized for Minkov-Kaasa
#
#  PDI = Power Distance, IDV = Individualism, MAS = Masculinity,
#  UAI = Uncertainty Avoidance, LTO = Long-Term Orientation,
#  IVR = Indulgence vs Restraint
#
#  IDV_COLL = Minkov-Kaasa Individualism-Collectivism (≈ Emancipation)
#  FLX_MON  = Minkov-Kaasa Flexibility-Monumentalism (≈ LTO reclassified)
#
#  The thesis values of IDV_COLL and FLX_MON were approximated from
#  Hofstede IDV/LTO; they are not the published Minkov-Kaasa scores.
# ═════════════════════════════════════════════════════════════════════

HOFSTEDE_CSV = os.path.join("data_raw", "culture", "hofstede_scores.csv")
HOFSTEDE_COLUMNS = [
    "country_iso3", "hofstede_pdi", "hofstede_idv", "hofstede_mas", "hofstede_uai",
    "hofstede_lto", "hofstede_ivr", "minkov_idv_coll", "minkov_flx_mon",
]


def get_hofstede_dataframe(path=HOFSTEDE_CSV):
    """Load Hofstede 6-D + Minkov-Kaasa scores from a local CSV.

    The scores are not redistributed with this repository. Download the
    6-D country scores from https://geerthofstede.com/research-and-vsm/dimension-data-matrix/
    (research use), add the Minkov & Kaasa (2022) columns, and save them to
    data_raw/culture/hofstede_scores.csv with the columns in HOFSTEDE_COLUMNS.
    Without the file the pipeline still runs; cultural features are NaN and
    the culture ablation block is effectively skipped.
    """
    if not os.path.exists(path):
        logger.warning(f"[CULTURE] {path} not found; cultural features will be NaN. "
                       "See DATA_SOURCES.md for how to obtain the scores.")
        return pd.DataFrame(columns=HOFSTEDE_COLUMNS)
    df = pd.read_csv(path)
    missing = set(HOFSTEDE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"hofstede_scores.csv is missing columns: {sorted(missing)}")
    return df[HOFSTEDE_COLUMNS]


def merge_cultural_features(panel, how="left"):
    """
    Merge cultural dimensions into the panel.
    Cultural values are static (country-level, not time-varying),
    so they join on country_iso3 only.

    Returns the panel with 8 new columns.
    Countries without Hofstede scores get NaN (handled by models that
    support missing values, e.g. HistGBR).
    """
    culture_df = get_hofstede_dataframe()
    n_before = len(panel)
    n_countries_before = panel["country_iso3"].nunique()

    merged = panel.merge(culture_df, on="country_iso3", how=how)

    n_matched = merged["hofstede_pdi"].notna().sum()
    coverage = n_matched / len(merged) * 100

    logger.info(f"[CULTURE] Merged cultural dimensions: {coverage:.0f}% row coverage "
                f"({culture_df.shape[0]} countries with scores)")

    return merged


def compute_cultural_composites(df):
    """
    Compute derived cultural composites that are more theoretically
    grounded than raw Hofstede dimensions.

    Inspired by Minkov & Kaasa (2022); the weights below are this
    project's choice, not taken from the paper:
      - Emancipation index: combines IDV_COLL with IVR
      - Long-term pragmatism: combines FLX_MON with LTO
      - Institutional trust proxy: inverse of UAI * PDI interaction

    These composites are designed to capture the "readiness for
    institutional change" that matters for policy adoption.
    """
    # Emancipation: high individualism + high indulgence → more open to change
    if "minkov_idv_coll" in df.columns and "hofstede_ivr" in df.columns:
        df["culture_emancipation"] = (
            0.6 * df["minkov_idv_coll"] / 100
            + 0.4 * df["hofstede_ivr"] / 100
        ).clip(0, 1)

    # Long-term pragmatism: high flexibility (FLX_MON runs in the flexibility
    # direction) + high LTO → patience for long-term policy
    if "minkov_flx_mon" in df.columns and "hofstede_lto" in df.columns:
        df["culture_pragmatism"] = (
            0.5 * df["minkov_flx_mon"] / 100
            + 0.5 * df["hofstede_lto"] / 100
        ).clip(0, 1)

    # Institutional trust proxy: low power distance + low uncertainty avoidance
    # → more willing to trust new institutional arrangements (like carbon pricing)
    if "hofstede_pdi" in df.columns and "hofstede_uai" in df.columns:
        df["culture_inst_trust"] = (
            1 - 0.5 * df["hofstede_pdi"] / 100
            - 0.5 * df["hofstede_uai"] / 100
        ).clip(0, 1)

    return df
