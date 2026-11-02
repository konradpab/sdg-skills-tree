#!/usr/bin/env python3
"""
Pre-flight check: validate everything needed before running on real data.

Usage:
    python preflight.py              # full check (environment + network)
    python preflight.py --env-only   # skip network checks
    python preflight.py --fix        # attempt to install missing packages

Exit codes:
    0 = all checks passed
    1 = fixable issues (missing optional packages, endpoint downtime)
    2 = blocking issues (wrong Python, missing core packages)
"""

import sys
import os
import importlib
import platform
import shutil
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent


# ── Formatting ───────────────────────────────────────────────────────

def _ok(msg):
    print(f"  \033[32m✓\033[0m {msg}")

def _warn(msg):
    print(f"  \033[33m⚠\033[0m {msg}")

def _fail(msg):
    print(f"  \033[31m✗\033[0m {msg}")

def _info(msg):
    print(f"  \033[36mℹ\033[0m {msg}")

def _header(msg):
    print(f"\n\033[1m{'─' * 60}\033[0m")
    print(f"\033[1m  {msg}\033[0m")
    print(f"\033[1m{'─' * 60}\033[0m")


# ── Checks ───────────────────────────────────────────────────────────

def check_python():
    """Verify Python version."""
    _header("Python version")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        _ok(f"Python {version_str}")
        return True
    else:
        _fail(f"Python {version_str} — need ≥ 3.10")
        return False


def check_packages():
    """Check all required Python packages."""
    _header("Python packages")
    required = {
        "pandas":       ("pandas",       ">=2.0"),
        "numpy":        ("numpy",        ">=1.24"),
        "pyarrow":      ("pyarrow",      ">=14.0"),
        "openpyxl":     ("openpyxl",     ">=3.1"),
        "sklearn":      ("scikit-learn", ">=1.3"),
        "xgboost":      ("xgboost",      ">=2.0"),
        "lightgbm":     ("lightgbm",     ">=4.0"),
        "shap":         ("shap",         ">=0.43"),
        "matplotlib":   ("matplotlib",   ">=3.7"),
        "seaborn":      ("seaborn",      ">=0.12"),
        "scipy":        ("scipy",        ">=1.11"),
        "networkx":     ("networkx",     ">=3.1"),
        "yaml":         ("pyyaml",       ">=6.0"),
        "jinja2":       ("jinja2",       ">=3.1"),
        "requests":     ("requests",     ">=2.31"),
    }

    all_ok = True
    missing = []
    for import_name, (pip_name, version) in required.items():
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            _ok(f"{pip_name} ({ver})")
        except ImportError:
            _fail(f"{pip_name} {version} — NOT INSTALLED")
            missing.append(pip_name)
            all_ok = False

    if missing:
        _info(f"Install missing: pip install {' '.join(missing)}")

    return all_ok, missing


def check_optional_packages():
    """Check optional but recommended packages."""
    _header("Optional packages")
    optional = {
        "geopandas":    "NUTS2 mapping",
        "plotly":       "Interactive visualizations",
    }
    for import_name, purpose in optional.items():
        try:
            importlib.import_module(import_name)
            _ok(f"{import_name} — {purpose}")
        except ImportError:
            _warn(f"{import_name} — not installed ({purpose})")


def check_quarto():
    """Check if Quarto is available."""
    _header("Quarto (optional)")
    if shutil.which("quarto"):
        import subprocess
        result = subprocess.run(["quarto", "--version"], capture_output=True, text=True)
        _ok(f"Quarto {result.stdout.strip()}")
        return True
    else:
        _warn("Quarto not installed — HTML report still works, PDF requires Quarto")
        _info("Install: https://quarto.org/docs/get-started/")
        return False


def check_project_structure():
    """Verify key project files exist."""
    _header("Project structure")
    required_files = [
        "config.yaml",
        "run_pipeline.py",
        "pyproject.toml",
        "registry/datasets.json",
        "registry/citations.bib",
        "src/ingest/real_data.py",
        "src/ingest/synthetic_data.py",
        "src/ingest/country_codes.py",
        "src/clean/normalize.py",
        "src/features/engineer.py",
        "src/models/train.py",
        "src/policy/planner.py",
        "src/viz/plots.py",
        "src/report/generate_report.py",
    ]
    all_ok = True
    for f in required_files:
        path = ROOT / f
        if path.exists():
            _ok(f)
        else:
            _fail(f"{f} — MISSING")
            all_ok = False

    # Check directories
    for d in ["data_raw", "data_processed", "models_saved", "outputs/recommendations", "reports/figures"]:
        path = ROOT / d
        if path.is_dir():
            _ok(f"{d}/")
        else:
            _warn(f"{d}/ — missing (will be created on first run)")
            path.mkdir(parents=True, exist_ok=True)

    return all_ok


def check_config():
    """Validate config.yaml."""
    _header("Configuration (config.yaml)")
    try:
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        # Check key sections
        for section in ["scope", "data_sources", "endpoints", "model", "planner"]:
            if section in cfg:
                _ok(f"Section: {section}")
            else:
                _fail(f"Section: {section} — MISSING")
                return False

        # Check enabled data sources have endpoints
        sources = cfg.get("data_sources", {})
        endpoints = cfg.get("endpoints", {})
        enabled = [s for s, v in sources.items() if v]
        _info(f"Enabled sources: {', '.join(enabled)}")

        for s in enabled:
            if s in endpoints or s in ("wdi",):  # wdi shares endpoint with wgi
                _ok(f"Endpoint configured: {s}")
            else:
                _warn(f"Endpoint missing for enabled source: {s}")

        return True
    except Exception as e:
        _fail(f"Cannot parse config.yaml: {e}")
        return False


def check_vdem():
    """Check for manually downloaded V-Dem data."""
    _header("V-Dem manual download")
    vdem_dir = ROOT / "data_raw" / "vdem"
    if vdem_dir.is_dir():
        csvs = list(vdem_dir.glob("*.csv"))
        if csvs:
            sizes = [(f.name, f.stat().st_size / 1e6) for f in csvs]
            for name, size_mb in sizes:
                if size_mb > 10:
                    _ok(f"Found: {name} ({size_mb:.0f} MB)")
                else:
                    _warn(f"Found: {name} ({size_mb:.1f} MB) — seems small, verify content")
            return True
        else:
            _warn("data_raw/vdem/ exists but no CSV files found")
    else:
        _warn("V-Dem data not found (optional — pipeline runs without it)")
        _info("Download from: https://v-dem.net/data/the-v-dem-dataset/")
        _info("Place CSV in: data_raw/vdem/")
    return False


def check_endpoints(timeout=10):
    """Test connectivity to all data endpoints."""
    _header("Endpoint connectivity")
    import yaml
    import requests

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    tests = [
        ("World Bank API",
         "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?date=2020&format=json&per_page=5",
         lambda r: isinstance(r.json(), list) and len(r.json()) >= 2),
        ("UNSD SDG API",
         "https://unstats.un.org/sdgapi/v1/sdg/Series/Data?seriesCode=EG_ELC_ACCS&pageSize=5",
         lambda r: "data" in r.json() or isinstance(r.json(), list)),
        ("Ember CSV",
         cfg["endpoints"]["ember"]["yearly_url"],
         lambda r: r.status_code == 200 and len(r.content) > 1000),
        ("Zenodo (PRIMAP)",
         "https://zenodo.org/api/records/10006301",
         lambda r: r.status_code == 200),
        ("SWIID",
         cfg["endpoints"]["swiid"]["download_url"],
         lambda r: r.status_code == 200),
        ("Carbon Pricing",
         cfg["endpoints"]["carbon_pricing"]["download_url"],
         lambda r: r.status_code == 200),
        ("UCDP GED",
         cfg["endpoints"]["ucdp"]["ged_url"],
         lambda r: r.status_code == 200),
    ]

    all_ok = True
    for name, url, validator in tests:
        try:
            # Use HEAD for large files to avoid downloading
            if url.endswith((".csv", ".zip", ".xlsx")):
                r = requests.head(url, timeout=timeout, allow_redirects=True)
                passed = r.status_code in (200, 302)
            else:
                r = requests.get(url, timeout=timeout)
                passed = validator(r)

            if passed:
                _ok(f"{name} — {r.status_code}")
            else:
                _warn(f"{name} — responded but validation failed (status {r.status_code})")
                all_ok = False
        except requests.exceptions.Timeout:
            _warn(f"{name} — timeout after {timeout}s (may be slow, not necessarily broken)")
            all_ok = False
        except requests.exceptions.ConnectionError as e:
            _fail(f"{name} — connection failed")
            _info(f"  URL: {url[:80]}...")
            all_ok = False
        except Exception as e:
            _fail(f"{name} — {type(e).__name__}: {e}")
            all_ok = False

    return all_ok


def check_disk_space():
    """Verify sufficient disk space."""
    _header("Disk space")
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / (1024 ** 3)
    if free_gb >= 5:
        _ok(f"{free_gb:.1f} GB free")
    elif free_gb >= 2:
        _warn(f"{free_gb:.1f} GB free — might be tight for full pipeline")
    else:
        _fail(f"{free_gb:.1f} GB free — need at least 2 GB")
    return free_gb >= 2


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-flight environment check")
    parser.add_argument("--env-only", action="store_true", help="Skip network endpoint checks")
    parser.add_argument("--fix", action="store_true", help="Attempt to install missing packages")
    args = parser.parse_args()

    print("\n\033[1m  SDG Skill-Trees — Pre-flight Check\033[0m")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Project:  {ROOT}")

    results = {}

    # Always run these
    results["python"] = check_python()
    pkg_ok, missing = check_packages()
    results["packages"] = pkg_ok
    check_optional_packages()
    results["quarto"] = check_quarto()
    results["structure"] = check_project_structure()
    results["config"] = check_config()
    results["vdem"] = check_vdem()
    results["disk"] = check_disk_space()

    # Network checks
    if not args.env_only:
        results["endpoints"] = check_endpoints()
    else:
        _info("Skipping endpoint checks (--env-only)")
        results["endpoints"] = None

    # Fix missing packages
    if args.fix and missing:
        _header("Auto-fix: installing missing packages")
        import subprocess
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        _info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)

    # Summary
    _header("SUMMARY")
    blocking = []
    warnings = []
    for check, passed in results.items():
        if passed is None:
            continue
        elif passed:
            _ok(check)
        elif check in ("python", "packages", "structure"):
            _fail(f"{check} — BLOCKING")
            blocking.append(check)
        else:
            _warn(f"{check} — non-blocking issue")
            warnings.append(check)

    print()
    if blocking:
        _fail(f"BLOCKED: Fix {', '.join(blocking)} before running the pipeline.")
        sys.exit(2)
    elif warnings:
        _warn(f"READY with warnings: {', '.join(warnings)}")
        _info("The pipeline will run but some data sources may be unavailable.")
        sys.exit(1)
    else:
        _ok("ALL CHECKS PASSED — ready to run: python run_pipeline.py --real")
        sys.exit(0)


if __name__ == "__main__":
    main()
