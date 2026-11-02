"""
Synthetic data generator for thesis pipeline demonstration.
Generates realistic country-year panel data for ~100 countries (1996-2023)
with correlated features that mirror known global distributions.

NOTE: This module exists because real API endpoints are unavailable in this
environment. The pipeline architecture is identical to what would run on
real data. Each synthetic series is documented with its generation logic.
"""

import numpy as np
import pandas as pd
import hashlib
import json
from datetime import datetime, timezone

np.random.seed(42)

# ─── Country metadata ───────────────────────────────────────────────
COUNTRIES = {
    # ISO3: (name, region, income_group, eu_member, lat, lon, base_gdppc, base_gov)
    "POL": ("Poland", "Europe & Central Asia", "High income", True, 52.0, 20.0, 12000, 0.65),
    "DEU": ("Germany", "Europe & Central Asia", "High income", True, 51.2, 10.5, 38000, 0.90),
    "NLD": ("Netherlands", "Europe & Central Asia", "High income", True, 52.1, 5.3, 42000, 0.92),
    "FRA": ("France", "Europe & Central Asia", "High income", True, 46.6, 2.3, 34000, 0.82),
    "ESP": ("Spain", "Europe & Central Asia", "High income", True, 40.5, -3.7, 26000, 0.75),
    "ITA": ("Italy", "Europe & Central Asia", "High income", True, 42.5, 12.6, 30000, 0.60),
    "SWE": ("Sweden", "Europe & Central Asia", "High income", True, 62.0, 15.0, 40000, 0.95),
    "ROU": ("Romania", "Europe & Central Asia", "Upper middle income", True, 46.0, 25.0, 6000, 0.40),
    "BGR": ("Bulgaria", "Europe & Central Asia", "Upper middle income", True, 42.7, 25.5, 5500, 0.35),
    "GRC": ("Greece", "Europe & Central Asia", "High income", True, 39.1, 21.8, 22000, 0.55),
    "GBR": ("United Kingdom", "Europe & Central Asia", "High income", False, 55.4, -3.4, 36000, 0.88),
    "NOR": ("Norway", "Europe & Central Asia", "High income", False, 60.5, 8.5, 55000, 0.95),
    "CHE": ("Switzerland", "Europe & Central Asia", "High income", False, 46.8, 8.2, 52000, 0.96),
    "USA": ("United States", "North America", "High income", False, 37.1, -95.7, 42000, 0.82),
    "CAN": ("Canada", "North America", "High income", False, 56.1, -106.3, 38000, 0.90),
    "JPN": ("Japan", "East Asia & Pacific", "High income", False, 36.2, 138.3, 35000, 0.85),
    "KOR": ("Korea, Rep.", "East Asia & Pacific", "High income", False, 35.9, 127.8, 18000, 0.72),
    "AUS": ("Australia", "East Asia & Pacific", "High income", False, -25.3, 133.8, 36000, 0.90),
    "CHN": ("China", "East Asia & Pacific", "Upper middle income", False, 35.9, 104.2, 2500, 0.30),
    "IND": ("India", "South Asia", "Lower middle income", False, 20.6, 79.0, 1500, 0.35),
    "BRA": ("Brazil", "Latin America & Caribbean", "Upper middle income", False, -14.2, -51.9, 8000, 0.40),
    "ZAF": ("South Africa", "Sub-Saharan Africa", "Upper middle income", False, -30.6, 22.9, 7000, 0.45),
    "NGA": ("Nigeria", "Sub-Saharan Africa", "Lower middle income", False, 9.1, 8.7, 1800, 0.15),
    "KEN": ("Kenya", "Sub-Saharan Africa", "Lower middle income", False, -0.02, 37.9, 1200, 0.25),
    "ETH": ("Ethiopia", "Sub-Saharan Africa", "Low income", False, 9.1, 40.5, 400, 0.18),
    "EGY": ("Egypt", "Middle East & North Africa", "Lower middle income", False, 26.8, 30.8, 3500, 0.30),
    "SAU": ("Saudi Arabia", "Middle East & North Africa", "High income", False, 23.9, 45.1, 22000, 0.35),
    "TUR": ("Turkey", "Europe & Central Asia", "Upper middle income", False, 38.9, 35.2, 9000, 0.40),
    "MEX": ("Mexico", "Latin America & Caribbean", "Upper middle income", False, 23.6, -102.6, 9000, 0.35),
    "IDN": ("Indonesia", "East Asia & Pacific", "Lower middle income", False, -0.8, 113.9, 2500, 0.30),
    "THA": ("Thailand", "East Asia & Pacific", "Upper middle income", False, 15.9, 100.5, 5000, 0.40),
    "VNM": ("Vietnam", "East Asia & Pacific", "Lower middle income", False, 14.1, 108.3, 1200, 0.28),
    "CHL": ("Chile", "Latin America & Caribbean", "High income", False, -35.7, -71.5, 10000, 0.72),
    "COL": ("Colombia", "Latin America & Caribbean", "Upper middle income", False, 4.6, -74.3, 5000, 0.35),
    "ARG": ("Argentina", "Latin America & Caribbean", "Upper middle income", False, -38.4, -63.6, 9000, 0.35),
    "MAR": ("Morocco", "Middle East & North Africa", "Lower middle income", False, 31.8, -7.1, 2800, 0.35),
    "TUN": ("Tunisia", "Middle East & North Africa", "Lower middle income", False, 33.9, 9.5, 3500, 0.38),
    "GHA": ("Ghana", "Sub-Saharan Africa", "Lower middle income", False, 7.9, -1.0, 1400, 0.40),
    "SEN": ("Senegal", "Sub-Saharan Africa", "Lower middle income", False, 14.5, -14.5, 1100, 0.38),
    "BGD": ("Bangladesh", "South Asia", "Lower middle income", False, 23.7, 90.4, 800, 0.22),
    "PAK": ("Pakistan", "South Asia", "Lower middle income", False, 30.4, 69.3, 1200, 0.20),
    "PHL": ("Philippines", "East Asia & Pacific", "Lower middle income", False, 12.9, 121.8, 2500, 0.35),
    "MYS": ("Malaysia", "East Asia & Pacific", "Upper middle income", False, 4.2, 101.9, 9000, 0.55),
    "PER": ("Peru", "Latin America & Caribbean", "Upper middle income", False, -9.2, -75.0, 4500, 0.35),
    "CZE": ("Czech Republic", "Europe & Central Asia", "High income", True, 49.8, 15.5, 15000, 0.72),
    "HUN": ("Hungary", "Europe & Central Asia", "High income", True, 47.2, 19.5, 12000, 0.62),
    "DNK": ("Denmark", "Europe & Central Asia", "High income", True, 56.3, 9.5, 42000, 0.96),
    "FIN": ("Finland", "Europe & Central Asia", "High income", True, 61.9, 25.7, 38000, 0.95),
    "PRT": ("Portugal", "Europe & Central Asia", "High income", True, 39.4, -8.2, 18000, 0.72),
    "AUT": ("Austria", "Europe & Central Asia", "High income", True, 47.5, 14.6, 38000, 0.88),
}

YEARS = list(range(1996, 2024))


def _trend(base, annual_growth, n_years, noise_std=0.02):
    """Generate a trending series with noise."""
    growth = np.cumprod(1 + annual_growth + np.random.normal(0, noise_std, n_years))
    return base * growth


def _bounded(series, lo=0, hi=100):
    return np.clip(series, lo, hi)


def _logistic_adoption(t, k=100, midpoint=15, steepness=0.3):
    """S-curve adoption pattern."""
    return k / (1 + np.exp(-steepness * (t - midpoint)))


def generate_country_data(iso3, meta, years):
    """Generate all indicators for one country across years."""
    name, region, income, eu, lat, lon, base_gdp, base_gov = meta
    n = len(years)
    t = np.arange(n)

    # Macro
    gdp_growth = 0.025 if base_gdp < 5000 else (0.018 if base_gdp < 15000 else 0.012)
    gdp_pc = _trend(base_gdp, gdp_growth, n, noise_std=0.015)
    pop_base = np.random.uniform(5e6, 200e6) if iso3 not in ["CHN", "IND", "USA"] else {"CHN": 1.3e9, "IND": 1.0e9, "USA": 2.8e8}[iso3]
    pop_growth = 0.02 if base_gdp < 3000 else (0.008 if base_gdp < 15000 else 0.003)
    population = _trend(pop_base, pop_growth, n, noise_std=0.002)
    urban_base = 30 + 40 * (base_gdp / 55000)
    urban_pct = _bounded(urban_base + t * np.random.uniform(0.2, 0.6), 15, 98)
    industry_gdp = np.random.uniform(18, 35) + np.random.normal(0, 1, n).cumsum() * 0.1
    industry_gdp = _bounded(industry_gdp, 8, 50)

    # Governance (slow-moving)
    gov_eff = _bounded(base_gov + t * np.random.uniform(-0.002, 0.008) + np.random.normal(0, 0.02, n), -0.5, 1.0)
    reg_quality = _bounded(gov_eff + np.random.normal(0, 0.05, n), -0.5, 1.0)
    rule_law = _bounded(gov_eff + np.random.normal(0, 0.04, n) - 0.02, -0.5, 1.0)
    corruption = _bounded(gov_eff + np.random.normal(0, 0.06, n) - 0.05, -0.5, 1.0)
    pol_stability = _bounded(base_gov - 0.1 + np.random.normal(0, 0.08, n), -1.0, 1.0)
    democracy = _bounded(0.3 + base_gov * 0.5 + np.random.normal(0, 0.03, n), 0, 1)

    # SDG7: Electricity access
    elec_access_base = min(99, 50 + 50 * (base_gdp / 20000))
    elec_access = _bounded(elec_access_base + t * np.random.uniform(0.3, 1.2), 20, 100)

    # SDG7: Renewables share (S-curve adoption)
    ren_base = np.random.uniform(5, 25) if iso3 not in ["NOR", "BRA"] else 60
    ren_boost = 0 if base_gov < 0.5 else np.random.uniform(0.5, 2.0)
    renewables_share = _bounded(ren_base + _logistic_adoption(t, k=30 + ren_boost * 10, midpoint=18, steepness=0.25), 0, 95)

    # SDG7: Energy intensity (declining)
    ei_base = 12 - 6 * (base_gdp / 55000) + np.random.uniform(-1, 1)
    energy_intensity = _bounded(ei_base * (1 - 0.015 * t + np.random.normal(0, 0.01, n).cumsum()), 1.5, 20)

    # SDG9 proxies
    rd_expenditure = _bounded(0.5 + 2.5 * (base_gdp / 55000) + t * 0.02 + np.random.normal(0, 0.05, n), 0.05, 5)
    manuf_va = _bounded(np.random.uniform(10, 25) + np.random.normal(0, 0.3, n).cumsum() * 0.05, 5, 35)

    # SDG13: Emissions
    co2_base = (base_gdp / 5000) * np.random.uniform(3, 12)
    co2_intensity_trend = -0.02 if base_gov > 0.6 else -0.008
    co2_pc = np.maximum(0.3, co2_base * np.exp(co2_intensity_trend * t + np.random.normal(0, 0.02, n).cumsum()))

    # Electricity mix
    coal_base = np.random.uniform(5, 60)
    if iso3 == "POL":
        coal_base = 80
    elif iso3 in ["FRA", "SWE"]:
        coal_base = 5
    coal_share = _bounded(coal_base - t * np.random.uniform(0.3, 1.5), 0, 95)
    gas_share = _bounded(np.random.uniform(10, 35) + np.random.normal(0, 0.5, n).cumsum() * 0.1, 0, 60)
    solar_share = _bounded(_logistic_adoption(t, k=np.random.uniform(8, 25), midpoint=20, steepness=0.35), 0, 40)
    wind_share = _bounded(_logistic_adoption(t, k=np.random.uniform(5, 30), midpoint=17, steepness=0.28), 0, 45)
    hydro_share = _bounded(np.random.uniform(5, 40) + np.random.normal(0, 0.3, n), 0, 70)
    nuclear_share = _bounded(np.random.uniform(0, 30) + np.random.normal(0, 0.2, n), 0, 50)
    # Normalize
    total = coal_share + gas_share + solar_share + wind_share + hydro_share + nuclear_share
    coal_share = coal_share / total * 100
    gas_share = gas_share / total * 100
    solar_share = solar_share / total * 100
    wind_share = wind_share / total * 100
    hydro_share = hydro_share / total * 100
    nuclear_share = nuclear_share / total * 100

    generation_twh = population / 1e6 * np.random.uniform(2, 12) * (1 + 0.02 * t)

    # Inequality
    gini_base = 55 - 20 * (base_gov) + np.random.uniform(-5, 5)
    gini_disp = _bounded(gini_base - t * np.random.uniform(-0.05, 0.15) + np.random.normal(0, 0.3, n), 22, 65)
    gini_mkt = _bounded(gini_disp + np.random.uniform(5, 15), 25, 70)

    # Carbon pricing
    has_carbon_price = (base_gov > 0.6) and (eu or np.random.random() > 0.5)
    if has_carbon_price:
        cp_start = np.random.randint(8, 20)
        carbon_price = np.where(t >= cp_start, np.random.uniform(5, 40) * (1 + 0.08 * (t - cp_start)), 0)
        carbon_coverage = np.where(t >= cp_start, _bounded(np.random.uniform(20, 60) + (t - cp_start) * 2, 0, 90), 0)
    else:
        carbon_price = np.zeros(n)
        carbon_coverage = np.zeros(n)

    # Conflict
    conflict_base = 0 if base_gov > 0.6 else np.random.uniform(0, 50)
    conflict_events = np.maximum(0, conflict_base + np.random.normal(0, conflict_base * 0.3, n))

    # Education
    edu_exp = _bounded(2 + 4 * (base_gdp / 55000) + np.random.normal(0, 0.1, n), 1, 8)

    # R&D / innovation
    patent_apps = np.maximum(0, (base_gdp / 1000) * np.random.uniform(5, 50) * (1 + 0.03 * t) + np.random.normal(0, 50, n))

    df = pd.DataFrame({
        "country_iso3": iso3,
        "year": years,
        # Macro
        "gdp_pc_ppp": gdp_pc,
        "population": population,
        "urban_pct": urban_pct,
        "industry_gdp_pct": industry_gdp,
        # Governance
        "gov_effectiveness": gov_eff,
        "regulatory_quality": reg_quality,
        "rule_of_law": rule_law,
        "control_corruption": corruption,
        "political_stability": pol_stability,
        "democracy_index": democracy,
        # SDG7
        "electricity_access_pct": elec_access,
        "renewables_share_pct": renewables_share,
        "energy_intensity_mj_usd": energy_intensity,
        # SDG9
        "rd_expenditure_gdp_pct": rd_expenditure,
        "manufacturing_va_gdp_pct": manuf_va,
        "patent_applications": patent_apps,
        # SDG13
        "co2_per_capita_t": co2_pc,
        # Electricity
        "coal_share_pct": coal_share,
        "gas_share_pct": gas_share,
        "solar_share_pct": solar_share,
        "wind_share_pct": wind_share,
        "hydro_share_pct": hydro_share,
        "nuclear_share_pct": nuclear_share,
        "generation_twh": generation_twh,
        # Inequality
        "gini_disposable": gini_disp,
        "gini_market": gini_mkt,
        # Policy
        "carbon_price_usd": carbon_price,
        "carbon_coverage_pct": carbon_coverage,
        # Conflict
        "conflict_events": conflict_events,
        # Education
        "education_exp_gdp_pct": edu_exp,
    })
    return df


def generate_dim_country():
    """Generate country dimension table."""
    rows = []
    for iso3, (name, region, income, eu, lat, lon, _, _) in COUNTRIES.items():
        rows.append({
            "country_iso3": iso3,
            "country_name": name,
            "region": region,
            "income_group": income,
            "eu_member": eu,
            "latitude": lat,
            "longitude": lon,
        })
    return pd.DataFrame(rows)


def generate_master_panel():
    """Generate full master panel for all countries and years."""
    frames = []
    for iso3, meta in COUNTRIES.items():
        frames.append(generate_country_data(iso3, meta, YEARS))
    panel = pd.concat(frames, ignore_index=True)

    # Introduce ~5% realistic missingness (more in poorer countries)
    for col in ["rd_expenditure_gdp_pct", "gini_disposable", "gini_market",
                "patent_applications", "democracy_index"]:
        mask = np.random.random(len(panel)) < 0.05
        panel.loc[mask, col] = np.nan

    # More missingness for early years
    early_mask = panel["year"] < 2000
    for col in ["solar_share_pct", "wind_share_pct", "carbon_price_usd", "carbon_coverage_pct"]:
        panel.loc[early_mask & (np.random.random(len(panel)) < 0.15), col] = np.nan

    return panel


def compute_hash(df):
    """Compute SHA256 hash of dataframe for reproducibility tracking."""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def run_ingestion(output_dir):
    """Main ingestion entry point."""
    print("[INGEST] Generating synthetic panel data...")
    dim_country = generate_dim_country()
    panel = generate_master_panel()

    dim_country.to_csv(f"{output_dir}/dim_country.csv", index=False)
    panel.to_csv(f"{output_dir}/master_panel_raw.csv", index=False)

    panel_hash = compute_hash(panel)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_countries": panel["country_iso3"].nunique(),
        "n_years": panel["year"].nunique(),
        "n_rows": len(panel),
        "n_cols": len(panel.columns),
        "hash_sha256": panel_hash,
        "note": "Synthetic data for pipeline demonstration. Structure mirrors real MVS datasets."
    }
    with open(f"{output_dir}/ingestion_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INGEST] Panel: {manifest['n_rows']} rows × {manifest['n_cols']} cols, "
          f"{manifest['n_countries']} countries, {manifest['n_years']} years")
    print(f"[INGEST] Hash: {panel_hash[:16]}...")
    return panel, dim_country


if __name__ == "__main__":
    run_ingestion("/home/claude/thesis/data_raw")
