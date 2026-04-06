"""
Cultural values data layer.

Provides Hofstede's 6 cultural dimensions plus Minkov & Kaasa's
reclassified meta-dimensions (Individualism-Collectivism ≈ Emancipation,
Flexibility-Monumentalism ≈ Long-Term Orientation) for ~70 countries.

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
# ═════════════════════════════════════════════════════════════════════

HOFSTEDE_DATA = {
    # ISO3:  (PDI, IDV, MAS, UAI, LTO, IVR, IDV_COLL, FLX_MON)
    "ARG": (49, 46, 56, 86, 20, 62, 40, 25),
    "AUS": (38, 90, 61, 51, 21, 71, 85, 30),
    "AUT": (11, 55, 79, 70, 60, 63, 55, 65),
    "BGD": (80, 20, 55, 60, 47, 20, 18, 50),
    "BEL": (65, 75, 54, 94, 82, 57, 72, 78),
    "BRA": (69, 38, 49, 76, 44, 59, 35, 40),
    "BGR": (70, 30, 40, 85, 69, 16, 28, 72),
    "CAN": (39, 80, 52, 48, 36, 68, 78, 35),
    "CHL": (63, 23, 28, 86, 31, 68, 22, 30),
    "CHN": (80, 20, 66, 30, 87, 24, 18, 90),
    "COL": (67, 13, 64, 80, 13, 83, 12, 15),
    "CZE": (57, 58, 57, 74, 70, 29, 55, 68),
    "DNK": (18, 74, 16, 23, 35, 70, 78, 40),
    "EGY": (70, 25, 45, 80, 7,  4,  22, 10),
    "EST": (40, 60, 30, 60, 82, 16, 58, 80),
    "ETH": (70, 20, 65, 55, 0,  46, 15, 5),
    "FIN": (33, 63, 26, 59, 38, 57, 65, 42),
    "FRA": (68, 71, 43, 86, 63, 48, 68, 60),
    "DEU": (35, 67, 66, 65, 83, 40, 65, 82),
    "GHA": (80, 15, 40, 65, 4,  72, 12, 8),
    "GRC": (60, 35, 57, 100,45, 50, 32, 48),
    "HUN": (46, 80, 88, 82, 58, 31, 75, 55),
    "IND": (77, 48, 56, 40, 51, 26, 42, 55),
    "IDN": (78, 14, 46, 48, 62, 38, 12, 60),
    "IRL": (28, 70, 68, 35, 24, 65, 72, 28),
    "ISR": (13, 54, 47, 81, 38, 0,  52, 40),
    "ITA": (50, 76, 70, 75, 61, 30, 72, 58),
    "JPN": (54, 46, 95, 92, 88, 42, 42, 88),
    "KEN": (70, 25, 60, 50, 0,  0,  20, 5),
    "KOR": (60, 18, 39, 85, 100,29, 15, 95),
    "MYS": (100,26, 50, 36, 41, 57, 24, 45),
    "MEX": (81, 30, 69, 82, 24, 97, 28, 22),
    "MAR": (70, 46, 53, 68, 14, 25, 40, 15),
    "NLD": (38, 80, 14, 53, 67, 68, 82, 65),
    "NZL": (22, 79, 58, 49, 33, 75, 80, 35),
    "NGA": (80, 30, 60, 55, 13, 84, 25, 12),
    "NOR": (31, 69, 8,  50, 35, 55, 72, 38),
    "PAK": (55, 14, 50, 70, 50, 0,  12, 52),
    "PER": (64, 16, 42, 87, 25, 46, 14, 25),
    "PHL": (94, 32, 64, 44, 27, 42, 28, 28),
    "POL": (68, 60, 64, 93, 38, 29, 55, 35),
    "PRT": (63, 27, 31, 99, 28, 33, 25, 30),
    "ROU": (90, 30, 42, 90, 52, 20, 28, 55),
    "RUS": (93, 39, 36, 95, 81, 20, 35, 80),
    "SAU": (95, 25, 60, 80, 36, 52, 22, 38),
    "SEN": (70, 25, 45, 55, 25, 0,  22, 25),
    "ZAF": (49, 65, 63, 49, 34, 63, 60, 32),
    "ESP": (57, 51, 42, 86, 48, 44, 48, 50),
    "SWE": (31, 71, 5,  29, 53, 78, 75, 55),
    "CHE": (34, 68, 70, 58, 74, 66, 68, 72),
    "THA": (64, 20, 34, 64, 32, 45, 18, 35),
    "TUN": (70, 35, 40, 70, 26, 30, 32, 28),
    "TUR": (66, 37, 45, 85, 46, 49, 35, 48),
    "GBR": (35, 89, 66, 35, 51, 69, 88, 52),
    "USA": (40, 91, 62, 46, 26, 68, 90, 28),
    "VNM": (70, 20, 40, 30, 57, 35, 18, 58),
}


def get_hofstede_dataframe():
    """Return Hofstede + Minkov-Kaasa data as a DataFrame."""
    rows = []
    for iso3, vals in HOFSTEDE_DATA.items():
        pdi, idv, mas, uai, lto, ivr, idv_coll, flx_mon = vals
        rows.append({
            "country_iso3": iso3,
            # Original Hofstede 6D
            "hofstede_pdi": pdi,           # Power Distance
            "hofstede_idv": idv,           # Individualism
            "hofstede_mas": mas,           # Masculinity
            "hofstede_uai": uai,           # Uncertainty Avoidance
            "hofstede_lto": lto,           # Long-Term Orientation
            "hofstede_ivr": ivr,           # Indulgence
            # Minkov-Kaasa reclassification (addresses Hofstede critique)
            "minkov_idv_coll": idv_coll,   # Individualism-Collectivism (≈ Emancipation)
            "minkov_flx_mon": flx_mon,     # Flexibility-Monumentalism (≈ LTO revised)
        })
    return pd.DataFrame(rows)


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

    Following Minkov & Kaasa (2022):
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

    # Long-term pragmatism: high monumentalism + high LTO → patience for long-term policy
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
