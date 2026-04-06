#!/usr/bin/env python3
"""
Test individual data sources in isolation.

Usage:
    python test_source.py wdi           # test one source
    python test_source.py wdi wgi       # test specific sources
    python test_source.py --all         # test all sources
    python test_source.py --list        # show available sources

Each test downloads a small sample, parses it, and shows what
the pipeline would get. No files are written to data_raw/.
"""

import sys
import os
import io
import argparse
import logging
import time
import yaml
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def test_worldbank(indicator_code, name, cfg):
    """Test a single World Bank indicator."""
    url = cfg["endpoints"]["wdi"]["api_url"].format(indicator=indicator_code)
    # Just fetch 5 rows to test
    url = url.replace("per_page=20000", "per_page=50").replace("date=1996:2023", "date=2020:2022")
    logger.info(f"  GET {url[:80]}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        logger.warning(f"  Unexpected response: {type(payload)}")
        return None
    records = payload[1] or []
    rows = []
    for rec in records:
        country = rec.get("country", {}).get("value", "?")
        year = rec.get("date")
        value = rec.get("value")
        if value is not None:
            rows.append({"country": country, "year": year, name: value})
    df = pd.DataFrame(rows)
    return df


def test_unsd_sdg(cfg):
    """Test UNSD SDG API with one series."""
    base = cfg["endpoints"]["unsd_sdg"]["base_url"]
    series = list(cfg["endpoints"]["unsd_sdg"]["series"].values())[0]  # first series
    url = f"{base}/Series/Data?seriesCode={series}&pageSize=20"
    logger.info(f"  GET {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    records = data.get("data", [])
    if not records:
        logger.warning(f"  No data in response. Keys: {list(data.keys())}")
        if isinstance(data, list):
            logger.info(f"  Response is a list of {len(data)} items")
            if data:
                logger.info(f"  First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
        return pd.DataFrame()
    rows = []
    for rec in records[:10]:
        rows.append({
            "geoAreaName": rec.get("geoAreaName", ""),
            "geoAreaCode": rec.get("geoAreaCode", ""),
            "year": rec.get("timePeriodStart", ""),
            "value": rec.get("value", ""),
        })
    return pd.DataFrame(rows)


def test_ember(cfg):
    """Test Ember by downloading just the first few KB."""
    url = cfg["endpoints"]["ember"]["yearly_url"]
    logger.info(f"  GET {url[:80]}...")
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    # Read only first 50KB to check format
    chunk = next(r.iter_content(50_000))
    r.close()
    df = pd.read_csv(io.BytesIO(chunk), nrows=20)
    return df


def test_primap(cfg):
    """Test PRIMAP by checking the Zenodo record exists."""
    url = cfg["endpoints"]["primap"]["download_url"]
    logger.info(f"  HEAD {url[:80]}...")
    r = requests.head(url, timeout=30, allow_redirects=True)
    logger.info(f"  Status: {r.status_code}, Content-Length: {r.headers.get('content-length', '?')}")
    if r.status_code == 200:
        size_mb = int(r.headers.get("content-length", 0)) / 1e6
        return pd.DataFrame([{"status": "OK", "size_mb": f"{size_mb:.1f}"}])
    return None


def test_swiid(cfg):
    """Test SWIID download URL."""
    url = cfg["endpoints"]["swiid"]["download_url"]
    logger.info(f"  HEAD {url}")
    r = requests.head(url, timeout=30, allow_redirects=True)
    logger.info(f"  Status: {r.status_code}")
    return pd.DataFrame([{"status": r.status_code, "url": url}])


def test_carbon_pricing(cfg):
    """Test Carbon Pricing Excel download."""
    url = cfg["endpoints"]["carbon_pricing"]["download_url"]
    logger.info(f"  HEAD {url[:80]}...")
    r = requests.head(url, timeout=30, allow_redirects=True)
    logger.info(f"  Status: {r.status_code}")
    return pd.DataFrame([{"status": r.status_code, "url": url[:60]}])


def test_ucdp(cfg):
    """Test UCDP GED download URL."""
    url = cfg["endpoints"]["ucdp"]["ged_url"]
    logger.info(f"  HEAD {url}")
    r = requests.head(url, timeout=30, allow_redirects=True)
    logger.info(f"  Status: {r.status_code}")
    return pd.DataFrame([{"status": r.status_code, "url": url}])


def test_vdem(cfg):
    """Check if V-Dem file is present locally."""
    vdem_dir = os.path.join(ROOT, "data_raw", "vdem")
    if os.path.isdir(vdem_dir):
        csvs = [f for f in os.listdir(vdem_dir) if f.endswith(".csv")]
        if csvs:
            path = os.path.join(vdem_dir, csvs[0])
            size_mb = os.path.getsize(path) / 1e6
            df = pd.read_csv(path, nrows=5, low_memory=False)
            return df
        else:
            logger.warning("  data_raw/vdem/ exists but no CSV found")
    else:
        logger.warning("  data_raw/vdem/ not found — V-Dem requires manual download")
    return None


SOURCES = {
    "wdi": ("World Bank WDI (macro indicators)", lambda cfg: test_worldbank("NY.GDP.PCAP.PP.KD", "gdp_pc_ppp", cfg)),
    "wgi": ("World Bank WGI (governance)", lambda cfg: test_worldbank("GE.EST", "gov_effectiveness", cfg)),
    "unsd_sdg": ("UNSD SDG API", test_unsd_sdg),
    "ember": ("Ember Electricity Data", test_ember),
    "primap": ("PRIMAP-hist Emissions (Zenodo)", test_primap),
    "swiid": ("SWIID Inequality", test_swiid),
    "carbon_pricing": ("Carbon Pricing Dashboard", test_carbon_pricing),
    "ucdp": ("UCDP GED Conflict", test_ucdp),
    "vdem": ("V-Dem Democracy (local file)", test_vdem),
}


def main():
    parser = argparse.ArgumentParser(description="Test individual data sources")
    parser.add_argument("sources", nargs="*", help="Source IDs to test")
    parser.add_argument("--all", action="store_true", help="Test all sources")
    parser.add_argument("--list", action="store_true", help="List available sources")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable data sources:")
        for sid, (desc, _) in SOURCES.items():
            print(f"  {sid:18s} {desc}")
        return

    cfg = load_config()

    sources_to_test = list(SOURCES.keys()) if args.all else (args.sources or ["wdi"])

    print(f"\n{'='*60}")
    print(f"  Testing {len(sources_to_test)} data source(s)")
    print(f"{'='*60}")

    results = {}
    for sid in sources_to_test:
        if sid not in SOURCES:
            print(f"\n  Unknown source: {sid}. Use --list to see options.")
            continue

        desc, test_fn = SOURCES[sid]
        print(f"\n{'─'*60}")
        print(f"  {sid}: {desc}")
        print(f"{'─'*60}")

        t0 = time.time()
        try:
            df = test_fn(cfg)
            elapsed = time.time() - t0
            if df is not None and not df.empty:
                print(f"\n  \033[32m✓ OK\033[0m ({elapsed:.1f}s)")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"\n  Sample (first 5 rows):")
                print(df.head().to_string(index=False))
                results[sid] = "OK"
            else:
                print(f"\n  \033[33m⚠ Empty result\033[0m ({elapsed:.1f}s)")
                results[sid] = "EMPTY"
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  \033[31m✗ FAILED\033[0m ({elapsed:.1f}s)")
            print(f"  Error: {type(e).__name__}: {e}")
            results[sid] = f"FAIL: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for sid, status in results.items():
        marker = "✓" if status == "OK" else "⚠" if status == "EMPTY" else "✗"
        color = "32" if status == "OK" else "33" if status == "EMPTY" else "31"
        print(f"  \033[{color}m{marker}\033[0m {sid}: {status}")


if __name__ == "__main__":
    main()
