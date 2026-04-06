"""
Real data ingestion for all MVS (Minimum Viable Stack) datasets.

Each function:
  1. Downloads from the documented endpoint
  2. Stores raw file + SHA-256 hash
  3. Parses into a tidy country_iso3 × year dataframe
  4. Returns the dataframe (normalization happens in clean/normalize.py)

If a download fails the function logs the error, saves a diagnostic file,
and returns an empty DataFrame so the pipeline can continue with partial data.
"""

import os
import io
import json
import hashlib
import zipfile
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd
import yaml

from src.ingest.country_codes import name_to_iso3, iso2_to_iso3

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────

def _load_config():
    """Load config.yaml from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / "config.yaml") as f:
        return yaml.safe_load(f)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save_raw(data: bytes, filename: str, raw_dir: str) -> str:
    """Save raw bytes to data_raw/ and return hash."""
    path = os.path.join(raw_dir, filename)
    os.makedirs(os.path.dirname(path) or raw_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    h = _hash_bytes(data)
    meta = {
        "filename": filename,
        "hash_sha256": h,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": len(data),
    }
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved {filename} ({len(data):,} bytes, sha256={h[:16]}...)")
    return h


def _get(url: str, params: dict = None, timeout: int = 120) -> requests.Response:
    """HTTP GET with retries and timeout."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/3 failed for {url}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise requests.RequestException(f"All retries failed for {url}")


def _empty_result(name: str, reason: str) -> pd.DataFrame:
    """Return empty frame and log skip."""
    logger.error(f"[{name}] SKIPPED — {reason}")
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════════════
#  1. UNSD SDG API
# ═════════════════════════════════════════════════════════════════════

def ingest_unsd_sdg(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Fetch SDG indicators via the UNSD SDG API (REST/JSON).
    Endpoint: https://unstats.un.org/sdgapi/v1/sdg/Series/{code}/GeoAreas
    Returns country-year panel with one column per indicator.
    """
    ep = cfg["endpoints"]["unsd_sdg"]
    base = ep["base_url"]
    page_size = ep.get("page_size", 1000)

    all_frames = []
    for our_name, series_code in ep["series"].items():
        logger.info(f"[UNSD] Fetching {our_name} ({series_code})...")
        try:
            # The SDG API uses: /v1/sdg/Series/Data?seriesCode=XX&pageSize=N
            url = f"{base}/Series/Data"
            params = {"seriesCode": series_code, "pageSize": page_size}
            resp = _get(url, params=params)
            payload = resp.json()

            rows = []
            # The response structure: { "data": [ { "geoAreaCode": "...", "timePeriodStart": "2020", "value": "..." } ] }
            for record in payload.get("data", []):
                geo_code = record.get("geoAreaCode", "")
                year_str = record.get("timePeriodStart", "")
                value_str = record.get("value", "")
                if not geo_code or not year_str:
                    continue
                try:
                    # geoAreaCode is the UN M49 numeric code; we need ISO3
                    # The API also provides geoAreaName which we can map
                    geo_name = record.get("geoAreaName", "")
                    iso3 = name_to_iso3(geo_name)
                    if iso3 is None:
                        continue
                    year = int(float(year_str))
                    value = float(value_str) if value_str not in ("", "N", "NA", "-") else np.nan
                    rows.append({"country_iso3": iso3, "year": year, our_name: value})
                except (ValueError, TypeError):
                    continue

            if rows:
                df = pd.DataFrame(rows)
                df = df.groupby(["country_iso3", "year"]).first().reset_index()
                all_frames.append(df)
                logger.info(f"  {our_name}: {len(df)} rows")

            _save_raw(resp.content, f"unsd_sdg/{series_code}.json", raw_dir)
        except Exception as e:
            logger.warning(f"[UNSD] Failed to fetch {our_name}: {e}")
            continue

    if not all_frames:
        return _empty_result("UNSD SDG", "no series fetched successfully")

    # Merge all series on country_iso3 × year
    result = all_frames[0]
    for df in all_frames[1:]:
        result = result.merge(df, on=["country_iso3", "year"], how="outer")

    # Rename to pipeline column names
    rename_map = {
        "elec_access": "electricity_access_pct",
        "renewables": "renewables_share_pct",
        "energy_int": "energy_intensity_mj_usd",
        "manuf_va": "manufacturing_va_gdp_pct",
        "rd_expend": "rd_expenditure_gdp_pct",
        "co2_pc": "co2_per_capita_t",
    }
    result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})
    return result


# ═════════════════════════════════════════════════════════════════════
#  2. Ember Yearly Electricity Data
# ═════════════════════════════════════════════════════════════════════

def ingest_ember(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Download Ember yearly electricity data (long-format CSV).
    Checks for a local file first (Ember blocks direct CSV downloads).
    Pivots into country-year with columns for each generation source.
    """
    ep = cfg["endpoints"]["ember"]

    # Check for manually downloaded local file first
    local_file = ep.get("local_file", "")
    local_path = os.path.join(raw_dir, local_file) if local_file else ""
    if local_path and os.path.exists(local_path):
        logger.info(f"[EMBER] Reading from local file: {local_path}")
        df = pd.read_csv(local_path, low_memory=False)
    else:
        # Try downloading (may fail with 403)
        url = ep["yearly_url"]
        logger.info(f"[EMBER] Downloading from {url}...")
        try:
            resp = _get(url, timeout=180)
            _save_raw(resp.content, "ember/yearly_full_release.csv", raw_dir)
            df = pd.read_csv(io.BytesIO(resp.content))
        except Exception as e:
            return _empty_result("Ember",
                f"{e}\n"
                f"  Ember blocks direct downloads. Manual download required:\n"
                f"  1. Go to https://ember-energy.org/data/yearly-electricity-data/\n"
                f"  2. Download the 'Yearly electricity data' CSV (long format)\n"
                f"  3. Place it in: {raw_dir}/ember/yearly_full_release_long_format.csv"
            )

    # Ember long-format has columns: Area, Year, Category, Subcategory, Variable, Unit, Value
    # We need country-level rows (filter out regions)
    df.columns = df.columns.str.strip().str.lower()

    # Map country names to ISO3
    if "country_code" in df.columns:
        # Some versions have ISO3 directly
        df["country_iso3"] = df["country_code"].str.strip().str.upper()
    elif "area" in df.columns:
        df["country_iso3"] = df["area"].apply(name_to_iso3)
    else:
        return _empty_result("Ember", "cannot find country column")

    df = df.dropna(subset=["country_iso3"])

    # Pivot key variables
    # Target variables: generation by source (TWh), demand, emissions
    var_col = "variable" if "variable" in df.columns else "series"
    val_col = "value" if "value" in df.columns else "generation"

    if var_col not in df.columns:
        return _empty_result("Ember", f"missing variable column (tried 'variable', 'series')")

    # Map Ember variable names to our column names
    ember_to_col = {
        "Coal": "coal_share_pct",
        "Gas": "gas_share_pct",
        "Solar": "solar_share_pct",
        "Wind": "wind_share_pct",
        "Hydro": "hydro_share_pct",
        "Nuclear": "nuclear_share_pct",
        "Electricity demand": "demand_twh",
        "Total generation": "generation_twh",
        "Power sector emissions": "power_emissions_mtco2",
    }

    # Filter to share-of-generation rows (unit = %)
    frames = []
    for ember_var, our_col in ember_to_col.items():
        mask = df[var_col].str.contains(ember_var, case=False, na=False)
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub = sub[["country_iso3", "year", val_col]].rename(columns={val_col: our_col})
        sub = sub.groupby(["country_iso3", "year"]).first().reset_index()
        frames.append(sub)

    if not frames:
        return _empty_result("Ember", "no matching variables found after filter")

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["country_iso3", "year"], how="outer")
    return result


# ═════════════════════════════════════════════════════════════════════
#  3. PRIMAP-hist Emissions
# ═════════════════════════════════════════════════════════════════════

def ingest_primap(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Download PRIMAP-hist from Zenodo. Wide-format CSV with year columns.
    Checks for a local file first.
    """
    ep = cfg["endpoints"]["primap"]

    # Check for local file first
    local_file = ep.get("local_file", "")
    local_path = os.path.join(raw_dir, local_file) if local_file else ""
    if local_path and os.path.exists(local_path):
        logger.info(f"[PRIMAP] Reading from local file: {local_path}")
        df = pd.read_csv(local_path, low_memory=False)
    else:
        url = ep["download_url"]
        logger.info(f"[PRIMAP] Downloading from Zenodo...")
        try:
            resp = _get(url, timeout=300)
            raw_bytes = resp.content
            # Handle zip files (some PRIMAP releases are zipped)
            if raw_bytes[:2] == b'PK':
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    csv_name = next((n for n in zf.namelist() if n.endswith('.csv')), None)
                    if csv_name is None:
                        return _empty_result("PRIMAP", "no CSV in zip")
                    with zf.open(csv_name) as csvf:
                        df = pd.read_csv(csvf, low_memory=False)
                _save_raw(raw_bytes, "primap/primap_hist.zip", raw_dir)
            else:
                _save_raw(raw_bytes, "primap/primap_hist.csv", raw_dir)
                df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            return _empty_result("PRIMAP", str(e))

    # PRIMAP has: source, scenario, country (ISO3), entity, unit, category, then year columns
    df.columns = df.columns.str.strip()

    # Find the ISO3 column
    iso_col = None
    for c in ["country (ISO3)", "area (ISO3)", "country"]:
        if c in df.columns:
            iso_col = c
            break
    if iso_col is None:
        return _empty_result("PRIMAP", "cannot find country ISO3 column")

    # Filter to CO2 and total GHG
    entity_col = "entity" if "entity" in df.columns else df.columns[3]

    # Get year columns (numeric)
    year_cols = [c for c in df.columns if c.isdigit() and 1990 <= int(c) <= 2025]

    results = []
    for entity_filter, our_col in [("CO2", "co2_mt"), ("KYOTOGHG", "ghg_mt_co2eq")]:
        sub = df[df[entity_col].str.contains(entity_filter, case=False, na=False)]
        if sub.empty:
            continue
        # Melt year columns to long format
        melted = sub.melt(
            id_vars=[iso_col],
            value_vars=year_cols,
            var_name="year",
            value_name=our_col,
        )
        melted = melted.rename(columns={iso_col: "country_iso3"})
        melted["year"] = melted["year"].astype(int)
        melted["country_iso3"] = melted["country_iso3"].str.strip().str.upper()
        melted = melted.groupby(["country_iso3", "year"]).first().reset_index()
        results.append(melted)

    if not results:
        return _empty_result("PRIMAP", "no CO2/GHG entities found")

    result = results[0]
    for r in results[1:]:
        result = result.merge(r, on=["country_iso3", "year"], how="outer")
    return result


# ═════════════════════════════════════════════════════════════════════
#  4. World Bank Indicators (WGI + WDI)
# ═════════════════════════════════════════════════════════════════════

def _ingest_worldbank_indicator(indicator_code: str, our_col: str, cfg: dict) -> pd.DataFrame:
    """Fetch a single World Bank indicator via API v2 (JSON)."""
    url_template = cfg["endpoints"]["wdi"]["api_url"]  # Same API for WGI and WDI
    url = url_template.format(indicator=indicator_code)
    logger.info(f"  Fetching {indicator_code} → {our_col}...")

    try:
        resp = _get(url, timeout=120)
        payload = resp.json()
    except Exception as e:
        logger.warning(f"  Failed: {e}")
        return pd.DataFrame()

    # World Bank API returns [metadata, data_array]
    if not isinstance(payload, list) or len(payload) < 2:
        logger.warning(f"  Unexpected response structure for {indicator_code}")
        return pd.DataFrame()

    records = payload[1] or []
    rows = []
    for rec in records:
        iso2 = rec.get("country", {}).get("id", "")
        iso3 = iso2_to_iso3(iso2)
        if iso3 is None:
            continue
        year = rec.get("date")
        value = rec.get("value")
        if year is None or value is None:
            continue
        try:
            rows.append({"country_iso3": iso3, "year": int(year), our_col: float(value)})
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["country_iso3", "year"]).first().reset_index()


def ingest_wgi(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """Fetch all 6 WGI indicators via World Bank API."""
    logger.info("[WGI] Fetching governance indicators...")
    frames = []
    for our_col, wb_code in cfg["endpoints"]["wgi"]["indicators"].items():
        df = _ingest_worldbank_indicator(wb_code, our_col, cfg)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)  # Be polite to the API

    if not frames:
        return _empty_result("WGI", "no indicators fetched")
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["country_iso3", "year"], how="outer")
    return result


def ingest_wdi(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """Fetch all WDI macro indicators via World Bank API."""
    logger.info("[WDI] Fetching macro indicators...")
    frames = []
    for our_col, wb_code in cfg["endpoints"]["wdi"]["indicators"].items():
        df = _ingest_worldbank_indicator(wb_code, our_col, cfg)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        return _empty_result("WDI", "no indicators fetched")
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["country_iso3", "year"], how="outer")
    return result


# ═════════════════════════════════════════════════════════════════════
#  5. V-Dem
# ═════════════════════════════════════════════════════════════════════

def ingest_vdem(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Read V-Dem CSV from data_raw/vdem/ (must be downloaded manually).
    V-Dem requires free registration at https://v-dem.net/data/the-v-dem-dataset/
    """
    vdem_dir = os.path.join(raw_dir, "vdem")
    candidates = [
        os.path.join(vdem_dir, f) for f in os.listdir(vdem_dir)
        if f.endswith(".csv")
    ] if os.path.isdir(vdem_dir) else []

    if not candidates:
        return _empty_result("V-Dem",
            "No CSV found in data_raw/vdem/. Download from https://v-dem.net/data/the-v-dem-dataset/ "
            "(free registration required) and place the Country-Year CSV there.")

    fpath = candidates[0]
    logger.info(f"[V-DEM] Reading {fpath}...")

    df = pd.read_csv(fpath, low_memory=False)
    target_vars = cfg["endpoints"]["vdem"]["variables"]
    keep_cols = ["country_text_id", "year"] + [v for v in target_vars if v in df.columns]
    df = df[keep_cols].copy()
    df = df.rename(columns={"country_text_id": "country_iso3"})

    # V-Dem uses ISO3 in country_text_id already
    # Rename variables to our names
    rename = {
        "v2x_polyarchy": "democracy_index",
        "v2x_libdem": "liberal_democracy",
        "v2x_partipdem": "participatory_democracy",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


# ═════════════════════════════════════════════════════════════════════
#  6. SWIID Inequality
# ═════════════════════════════════════════════════════════════════════

def ingest_swiid(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Download SWIID zip from fsolt.org, extract swiid_summary.csv.
    """
    url = cfg["endpoints"]["swiid"]["download_url"]
    logger.info(f"[SWIID] Downloading from {url}...")

    try:
        resp = _get(url, timeout=120)
        _save_raw(resp.content, "swiid/swiid.zip", raw_dir)

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the summary CSV
            csv_name = None
            for name in zf.namelist():
                if "summary" in name.lower() and name.endswith(".csv"):
                    csv_name = name
                    break
            if csv_name is None:
                # Fallback: any CSV
                csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if csv_name is None:
                return _empty_result("SWIID", "no CSV found in zip")

            with zf.open(csv_name) as csvf:
                df = pd.read_csv(csvf)
    except Exception as e:
        return _empty_result("SWIID", str(e))

    df.columns = df.columns.str.strip().str.lower()

    # SWIID has: country, year, gini_disp, gini_disp_se, gini_mkt, gini_mkt_se
    if "country" in df.columns:
        df["country_iso3"] = df["country"].apply(name_to_iso3)
    elif "country_code" in df.columns:
        df["country_iso3"] = df["country_code"].str.upper()
    else:
        return _empty_result("SWIID", "no country column found")

    df = df.dropna(subset=["country_iso3"])
    keep = ["country_iso3", "year"]
    rename = {}
    for col in ["gini_disp", "gini_mkt"]:
        if col in df.columns:
            rename[col] = {"gini_disp": "gini_disposable", "gini_mkt": "gini_market"}[col]
            keep.append(col)

    df = df[keep].rename(columns=rename)
    return df


# ═════════════════════════════════════════════════════════════════════
#  7. Carbon Pricing Dashboard
# ═════════════════════════════════════════════════════════════════════

def ingest_carbon_pricing(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Download the World Bank Carbon Pricing Dashboard Excel file.
    """
    url = cfg["endpoints"]["carbon_pricing"]["download_url"]
    logger.info(f"[CARBON] Downloading from {url}...")

    try:
        resp = _get(url, timeout=120)
        _save_raw(resp.content, "carbon_pricing/data-latest.xlsx", raw_dir)
        # Try reading all sheets to find the data
        xls = pd.ExcelFile(io.BytesIO(resp.content))
        # Usually the first sheet has the data
        df = pd.read_excel(xls, sheet_name=0)
    except Exception as e:
        return _empty_result("Carbon Pricing", str(e))

    df.columns = df.columns.str.strip().str.lower()

    # Map country/jurisdiction to ISO3
    country_col = next((c for c in df.columns if "country" in c or "jurisdiction" in c), None)
    if country_col is None:
        return _empty_result("Carbon Pricing", "no country/jurisdiction column")

    df["country_iso3"] = df[country_col].apply(name_to_iso3)
    df = df.dropna(subset=["country_iso3"])

    # Extract price and coverage columns
    year_col = next((c for c in df.columns if "year" in c), None)
    price_col = next((c for c in df.columns if "price" in c), None)
    coverage_col = next((c for c in df.columns if "coverage" in c), None)

    result_cols = ["country_iso3"]
    if year_col:
        result_cols.append(year_col)
    rename = {}
    if price_col:
        result_cols.append(price_col)
        rename[price_col] = "carbon_price_usd"
    if coverage_col:
        result_cols.append(coverage_col)
        rename[coverage_col] = "carbon_coverage_pct"

    df = df[result_cols].rename(columns=rename)
    if year_col and year_col != "year":
        df = df.rename(columns={year_col: "year"})
    return df


# ═════════════════════════════════════════════════════════════════════
#  8. UCDP GED (Conflict)
# ═════════════════════════════════════════════════════════════════════

def ingest_ucdp(raw_dir: str, cfg: dict) -> pd.DataFrame:
    """
    Download UCDP GED CSV (zipped), aggregate events to country-year.
    """
    url = cfg["endpoints"]["ucdp"]["ged_url"]
    logger.info(f"[UCDP] Downloading GED from {url}...")

    try:
        resp = _get(url, timeout=180)
        _save_raw(resp.content, "ucdp/ged.zip", raw_dir)

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if csv_name is None:
                return _empty_result("UCDP", "no CSV in zip")
            with zf.open(csv_name) as csvf:
                df = pd.read_csv(csvf, low_memory=False)
    except Exception as e:
        return _empty_result("UCDP", str(e))

    df.columns = df.columns.str.strip().str.lower()

    # UCDP GED has: country, country_id, year, best (best estimate fatalities), ...
    country_col = next((c for c in df.columns if c in ("country", "country_name")), None)
    iso_col = next((c for c in df.columns if "country_id" in c or "gwno" in c), None)

    if country_col:
        df["country_iso3"] = df[country_col].apply(name_to_iso3)
    elif iso_col:
        df["country_iso3"] = df[iso_col].apply(lambda x: name_to_iso3(str(x)))
    else:
        return _empty_result("UCDP", "no country column found")

    df = df.dropna(subset=["country_iso3"])

    # Aggregate to country-year
    fatality_col = next((c for c in df.columns if c in ("best", "best_est", "deaths_best")), "best")
    agg = df.groupby(["country_iso3", "year"]).agg(
        conflict_events=("country_iso3", "count"),
        battle_deaths=(fatality_col, "sum") if fatality_col in df.columns else ("country_iso3", "count"),
    ).reset_index()

    return agg


# ═════════════════════════════════════════════════════════════════════
#  Master ingestion orchestrator
# ═════════════════════════════════════════════════════════════════════

INGEST_REGISTRY = {
    "unsd_sdg":       ingest_unsd_sdg,
    "ember":          ingest_ember,
    "primap":         ingest_primap,
    "wgi":            ingest_wgi,
    "wdi":            ingest_wdi,
    "vdem":           ingest_vdem,
    "swiid":          ingest_swiid,
    "carbon_pricing": ingest_carbon_pricing,
    "ucdp":           ingest_ucdp,
}


def run_ingestion_real(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all enabled ingestion functions, merge into master panel.
    Returns (panel, dim_country).
    """
    cfg = _load_config()
    source_toggles = cfg.get("data_sources", {})

    logger.info("=" * 60)
    logger.info("REAL DATA INGESTION")
    logger.info("=" * 60)

    frames = {}
    for source_id, ingest_fn in INGEST_REGISTRY.items():
        if not source_toggles.get(source_id, False):
            logger.info(f"[SKIP] {source_id} disabled in config.yaml")
            continue
        try:
            df = ingest_fn(raw_dir, cfg)
            if not df.empty:
                frames[source_id] = df
                logger.info(f"[OK] {source_id}: {len(df)} rows, cols={list(df.columns)}")
            else:
                logger.warning(f"[EMPTY] {source_id} returned empty")
        except Exception as e:
            logger.error(f"[FAIL] {source_id}: {e}")

    if not frames:
        raise RuntimeError("No datasets were successfully ingested. Check logs and network.")

    # Merge all frames on country_iso3 × year
    logger.info("Merging all sources into master panel...")
    merged = None
    for source_id, df in frames.items():
        df = df.copy()
        # Ensure types
        df["country_iso3"] = df["country_iso3"].astype(str).str.strip().str.upper()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"])

        if merged is None:
            merged = df
        else:
            merge_cols = ["country_iso3", "year"]
            # Avoid duplicate columns
            overlap = set(merged.columns) & set(df.columns) - set(merge_cols)
            if overlap:
                logger.warning(f"  Column overlap from {source_id}: {overlap} — keeping first")
                df = df.drop(columns=list(overlap), errors="ignore")
            merged = merged.merge(df, on=merge_cols, how="outer")

    # Filter to configured time range
    scope = cfg.get("scope", {})
    year_min = scope.get("train_start", 1996)
    year_max = scope.get("train_end", 2023)
    merged = merged[(merged["year"] >= year_min) & (merged["year"] <= year_max)]

    # Build dim_country from the merged data
    dim_country = (
        merged[["country_iso3"]].drop_duplicates()
        .assign(country_name="", region="", income_group="", eu_member=False)
    )

    # Save manifest
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_countries": int(merged["country_iso3"].nunique()),
        "n_years": int(merged["year"].nunique()),
        "n_rows": len(merged),
        "n_cols": len(merged.columns),
        "sources_ingested": list(frames.keys()),
        "sources_skipped": [s for s in INGEST_REGISTRY if s not in frames],
    }
    with open(os.path.join(raw_dir, "ingestion_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Master panel: {manifest['n_rows']} rows × {manifest['n_cols']} cols, "
                f"{manifest['n_countries']} countries")
    return merged, dim_country
