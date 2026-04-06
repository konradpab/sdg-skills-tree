# Deployment Guide — Running on Real Data

This guide covers everything from zero to a working pipeline on real-world data:
environment setup, dependency management, endpoint validation, execution, 
troubleshooting, and production hardening.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Choose Your Environment](#2-choose-your-environment)
3. [Environment Setup (Step by Step)](#3-environment-setup)
4. [Pre-flight Validation](#4-pre-flight-validation)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [What Happens at Each Stage](#6-what-happens-at-each-stage)
7. [Manual Data Steps](#7-manual-data-steps)
8. [Endpoint Reference & Fallbacks](#8-endpoint-reference--fallbacks)
9. [Configuration Tuning](#9-configuration-tuning)
10. [Troubleshooting](#10-troubleshooting)
11. [Production Hardening](#11-production-hardening)
12. [Extending the Pipeline](#12-extending-the-pipeline)

---

## 1. Architecture Overview

The pipeline runs in seven sequential stages:

```
INGEST → NORMALIZE → FEATURES → TRAIN → RECOMMEND → VISUALIZE → REPORT
```

**Ingest** downloads 9 open datasets from public APIs/URLs and stores raw files
with SHA-256 hashes. **Normalize** enforces ISO3 country codes, clips outliers,
interpolates short gaps, and produces a coverage report. **Features** creates
lags, deltas, rolling means, and composite indices (capability, feasibility,
green transition). **Train** fits 5 model families per target with time-aware
splits and adds quantile + conformal uncertainty. **Recommend** scores 13
interventions through a multi-objective utility function respecting a DAG of
prerequisites. **Visualize** generates all report figures. **Report** produces
a self-contained HTML document with embedded images.

**Key files:**

| File | Role |
|------|------|
| `config.yaml` | Single source of truth for all parameters |
| `run_pipeline.py` | Main entry point (modes: `--real`, `--synthetic`, `--from-cache`) |
| `preflight.py` | Environment + endpoint health checker |
| `src/ingest/real_data.py` | All 9 dataset download & parse functions |
| `src/ingest/country_codes.py` | ISO3 mapping for ~150 country name variants |
| `src/clean/normalize.py` | Outlier clipping, interpolation, coverage |
| `src/features/engineer.py` | Feature construction |
| `src/models/train.py` | Model training + uncertainty |
| `src/policy/planner.py` | Intervention graph + recommendation engine |

---

## 2. Choose Your Environment

### Option A: Local Machine (fastest to start)

**Best for:** Development, iteration, thesis writing.

Requirements: Python 3.10+, 8 GB RAM, internet access, 5 GB disk.
Works on macOS, Linux, and WSL2 on Windows.

### Option B: Cloud VM (recommended for final runs)

**Best for:** Reproducible final run, reliable API access, long-running jobs.

Any cloud provider works. Recommended spec:

| Provider | Instance | Cost |
|----------|----------|------|
| GCP | `e2-standard-2` (2 vCPU, 8 GB) | ~$0.07/hr |
| AWS | `t3.large` (2 vCPU, 8 GB) | ~$0.08/hr |
| Azure | `B2s` (2 vCPU, 4 GB) | ~$0.04/hr |
| Hetzner | `CX21` (2 vCPU, 4 GB) | €5.29/mo |

Use Ubuntu 22.04 or 24.04 LTS. Total runtime is ~10-15 minutes.

### Option C: GitHub Codespaces (zero local setup)

Push the repo to GitHub, open in Codespaces (free tier: 120 core-hours/month).
The default 4-core instance is sufficient.

### Option D: Docker (maximum reproducibility)

**Best for:** Thesis submission, sharing with reviewers, CI/CD.

Requires Docker installed. No Python setup needed on the host.

---

## 3. Environment Setup

### 3.1 Local / Cloud VM Setup

```bash
# 1. System prerequisites (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git

# On macOS (with Homebrew):
# brew install python@3.12 git

# 2. Clone the repo
git clone https://github.com/konradpab/sdg-skill-trees.git
cd sdg-skill-trees

# 3. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -e .

# For development (linting + testing):
# pip install -e ".[dev]"

# For everything including extensions:
# pip install -e ".[dev,quarto,extensions]"

# 5. Verify installation
python preflight.py --env-only
```

### 3.2 Docker Setup

```bash
# Build the image
docker build -t sdg-skill-trees .

# Run the full pipeline (outputs persist via volume mounts)
docker run -v $(pwd)/reports:/thesis/reports \
           -v $(pwd)/outputs:/thesis/outputs \
           -v $(pwd)/data_processed:/thesis/data_processed \
           sdg-skill-trees --real

# Or use docker compose
docker compose up pipeline
```

### 3.3 What Gets Installed

Core stack (~400 MB):

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data manipulation |
| pyarrow | ≥14.0 | Parquet I/O |
| openpyxl | ≥3.1 | Excel file reading |
| scikit-learn | ≥1.3 | ML baselines, random forest, GBR |
| xgboost | ≥2.0 | Gradient boosting (alternative) |
| lightgbm | ≥4.0 | Gradient boosting (alternative) |
| shap | ≥0.43 | Feature importance / interpretability |
| matplotlib | ≥3.7 | Static figures |
| seaborn | ≥0.12 | Statistical plots |
| scipy | ≥1.11 | Statistics (Spearman correlation, etc.) |
| networkx | ≥3.1 | Intervention DAG |
| requests | ≥2.31 | HTTP downloads |
| pyyaml | ≥6.0 | Config parsing |
| jinja2 | ≥3.1 | HTML report templating |

---

## 4. Pre-flight Validation

Before running on real data, validate everything:

```bash
# Full check (environment + network endpoints)
python preflight.py

# Environment only (skip network — e.g. on a plane)
python preflight.py --env-only

# Attempt to auto-install missing packages
python preflight.py --fix
```

The script checks:
- Python version (≥3.10)
- All 15 required packages
- Project file structure
- config.yaml validity
- V-Dem manual download presence
- Disk space (≥2 GB)
- Connectivity to all 7 API/download endpoints

Exit codes: 0 = ready, 1 = warnings (non-blocking), 2 = blocking issues.

---

## 5. Running the Pipeline

### 5.1 First Run (Real Data)

```bash
python run_pipeline.py --real
```

This takes **3-15 minutes** depending on API response times. It downloads
all 9 datasets, builds the master panel, trains models, generates
recommendations for Poland/Germany/India, and produces `reports/final_report.html`.

### 5.2 Subsequent Runs (From Cache)

After a successful real-data run, skip re-downloading:

```bash
python run_pipeline.py --from-cache
```

### 5.3 Fast Iteration (Skip Models Too)

When iterating on the report or planner:

```bash
python run_pipeline.py --from-cache --skip-models
```

### 5.4 Different Case Studies

```bash
python run_pipeline.py --real --countries POL DEU BRA
python run_pipeline.py --from-cache --countries NGA ZAF KEN
```

### 5.5 Synthetic Mode (Testing)

```bash
python run_pipeline.py --synthetic
```

No network required. Runs the identical pipeline on synthetic data.
Use this to verify your environment works before switching to real data.

### 5.6 Makefile Shortcuts

```bash
make run               # Real data
make run-synthetic     # Synthetic
make run-cached        # From cache
make run-fast          # Cache + skip models
make test              # Run pytest
make clean             # Remove outputs (keep raw data)
make clean-all         # Remove everything
```

---

## 6. What Happens at Each Stage

### Stage 1: Ingest

For each enabled source in `config.yaml → data_sources`:

1. HTTP GET to the endpoint (with 3 retries, exponential backoff)
2. Save raw response bytes to `data_raw/<source>/` with `.meta.json` sidecar
3. Parse into a DataFrame with `country_iso3` and `year` columns
4. Outer-join all DataFrames into a single raw panel
5. Save `data_raw/master_panel_raw.csv` + `ingestion_manifest.json`

If any single source fails, the pipeline continues with the remaining sources
and logs a warning. You can later re-enable the failed source and re-run.

### Stage 2: Normalize

- Enforce ISO3 uppercase + integer year
- Remove aggregate rows (WLD, EUU, etc.)
- Clip values outside valid ranges to NaN (ranges defined in `normalize.py`)
- Produce `data_interim/coverage_report.json`
- Linearly interpolate gaps ≤2 years within each country
- Deduplicate

### Stage 3: Features

From ~30 raw columns, produces ~160 features:
- Lags: t-1, t-2 for all key indicators
- Deltas: Δ1y, Δ3y (backward-looking change)
- Rolling: 3-year rolling mean
- Composites: capability_index, feasibility_index, green_transition_index
- Log-transforms for skewed variables
- Forward targets: target_*_delta1y, target_*_delta3y

### Stage 4: Train

Time-aware splits (no leakage):
- Train: ≤2017 (~1100 rows)
- Validation: 2018-2020 (~150 rows)
- Test: 2021+ (~150 rows)

Per target variable (10 targets = 5 indicators × 2 horizons):
- 5 models: naive persistence, elastic net, random forest, GBR, HistGBR
- Best model selected by validation MAE
- Quantile regression at {10%, 25%, 50%, 75%, 90%}
- Conformal prediction calibrated on validation set (90% coverage)

### Stage 5: Recommend

For each case study country:
- Extract latest state vector
- Score 13 interventions via 6-component utility function
- Enforce DAG prerequisites via topological ordering
- Output ranked roadmap with utility scores, effect estimates, feasibility

### Stage 6-7: Visualize + Report

Generate 9 PNG figures and a self-contained HTML report with:
- Data coverage heatmap
- Model comparison bar charts
- Feature importance plots
- Country trajectory lines
- Intervention roadmap Gantt charts
- Full metrics tables
- Case study snapshots

---

## 7. Manual Data Steps

### V-Dem (Required for Democracy Indices)

V-Dem requires free registration. The pipeline runs without it, but you lose
`democracy_index`, `liberal_democracy`, and `participatory_democracy`.

**Steps:**

1. Go to https://v-dem.net/data/the-v-dem-dataset/
2. Click "Download" → register (free academic account)
3. Download **"Country-Year: V-Dem Core"** (CSV format, ~200 MB)
4. Place the CSV in `data_raw/vdem/`
5. Verify: `make vdem-check` or `ls data_raw/vdem/*.csv`

**If you skip V-Dem:** Set `vdem: false` in `config.yaml → data_sources`.
The governance dimension is still covered by WGI (6 indicators).

### World Values Survey (Optional Extension)

WVS also requires registration and a license agreement:
https://www.worldvaluessurvey.org/AJDownloadLicense.jsp

This is not in the MVS and is off by default. If you want culture/values
features, download the longitudinal data file and write an ingestion function
following the pattern in `src/ingest/real_data.py`.

---

## 8. Endpoint Reference & Fallbacks

Every data source has a primary endpoint and a fallback procedure if it
breaks. URLs change — this section documents how to fix each one.

### World Bank API (WDI + WGI)

**Primary:** `https://api.worldbank.org/v2/country/all/indicator/{code}?date=1996:2023&format=json&per_page=20000`

**Stability:** Very stable. Rarely changes. Occasionally returns empty for
the most recent year (normal — data hasn't been published yet).

**If it breaks:**
- Test manually: `curl "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?date=2020&format=json&per_page=5"`
- Check indicator codes at https://data.worldbank.org/indicator
- If the API is down, download bulk CSVs from https://databank.worldbank.org/

**Config:** `config.yaml → endpoints.wgi.indicators` and `endpoints.wdi.indicators`

### UNSD SDG API

**Primary:** `https://unstats.un.org/sdgapi/v1/sdg/Series/Data?seriesCode={code}&pageSize=1000`

**Stability:** Moderate. The API has been restructured several times. Response
format may change.

**If it breaks:**
- Check current schema at https://unstats.un.org/sdgapi/swagger/
- Test: `curl "https://unstats.un.org/sdgapi/v1/sdg/Series/Data?seriesCode=EG_ELC_ACCS&pageSize=5"`
- If the response structure changed, update `ingest_unsd_sdg()` in `real_data.py`
- Fallback: download SDG data manually from https://unstats.un.org/sdgs/dataportal

**Config:** `config.yaml → endpoints.unsd_sdg.series`

### Ember Electricity

**Primary:** `https://ember-energy.org/app/uploads/2024/09/yearly_full_release_long_format.csv`

**Stability:** URL changes with each annual release (the date in the path changes).

**If it 404s:**
1. Go to https://ember-energy.org/data/yearly-electricity-data/
2. Click "Download" and copy the new CSV URL
3. Update `config.yaml → endpoints.ember.yearly_url`
4. Or download manually and place in `data_raw/ember/yearly_full_release.csv`

**Column format:** Ember occasionally restructures columns. If parsing fails,
check column names in the downloaded CSV against `ingest_ember()`.

### PRIMAP-hist (Zenodo)

**Primary:** `https://zenodo.org/records/10006301/files/Guetschow_et_al_2023-PRIMAP-hist_v2.5.1_final_no_extrap_15-Oct-2023.csv`

**Stability:** URL changes with each new Zenodo record (new version = new DOI).

**If it 404s:**
1. Search https://zenodo.org/search?q=primap-hist for the latest record
2. Copy the direct CSV download link
3. Update `config.yaml → endpoints.primap.download_url`

### SWIID

**Primary:** `https://fsolt.org/swiid/swiid9_6.zip`

**Stability:** URL changes with version bumps (9_6 → 9_7 etc.).

**If it 404s:**
1. Check https://fsolt.org/swiid/ for current version
2. Update `config.yaml → endpoints.swiid.download_url`

### Carbon Pricing Dashboard

**Primary:** `https://carbonpricingdashboard.worldbank.org/sites/default/files/data-latest.xlsx`

**Stability:** Generally stable (the `data-latest.xlsx` endpoint auto-updates).

**If it breaks:** Download manually from
https://carbonpricingdashboard.worldbank.org/ and place in
`data_raw/carbon_pricing/data-latest.xlsx`.

### UCDP GED

**Primary:** `https://ucdp.uu.se/downloads/ged/ged241-csv.zip`

**Stability:** URL changes with each annual release (ged241 → ged251 etc.).

**If it 404s:**
1. Check https://ucdp.uu.se/downloads/ for the current GED version
2. Update `config.yaml → endpoints.ucdp.ged_url`

---

## 9. Configuration Tuning

All parameters live in `config.yaml`. Key adjustments:

### Change Time Splits

```yaml
model:
  train_cutoff: 2018    # push forward if you have more recent data
  val_cutoff: 2021
```

### Change Forecast Horizons

```yaml
scope:
  forecast_horizons: [1, 3, 5]   # add 5-year forecasts
```

Then update `src/features/engineer.py:add_forward_targets()` and
`src/models/train.py:TARGET_NAMES` to include the new horizon.

### Add/Remove Targets

```yaml
model:
  targets:
    - "renewables_share_pct"
    - "co2_per_capita_t"
    - "energy_intensity_mj_usd"
    - "electricity_access_pct"
    - "gini_disposable"
    - "coal_share_pct"          # add this
```

### Adjust Utility Weights

```yaml
planner:
  weights:
    w_sdg:         0.20     # reduce SDG weight
    w_co2:         0.40     # increase CO2 priority
    w_equity:      0.15
    w_feasibility: 0.10
    w_speed:       0.10
    w_cost:        0.05
```

### Disable a Problematic Source

```yaml
data_sources:
  unsd_sdg: false    # skip if API is down
```

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"

```bash
pip install xgboost lightgbm shap pyarrow openpyxl
```

Or reinstall everything: `pip install -e .`

### "Connection refused" / "Temporary failure in name resolution"

Your network blocks outbound HTTP. Try:
1. Different network (home vs institutional)
2. A cloud VM
3. VPN

After one successful download, use `--from-cache` for all subsequent runs.

### "No CSV found in zip" (SWIID or UCDP)

The download URL changed. Check the data source website for the current link
and update `config.yaml`. See Section 8 for each source.

### "Empty panel after merge"

Check `pipeline.log` for warnings about unmapped country names. The
`country_codes.py` module covers ~150 name variants, but new datasets may
use names not yet mapped. Add missing mappings to `_NAME_TO_ISO3` in
`src/ingest/country_codes.py`.

### "All targets have val_mae of 0 or NaN"

Usually means the train/test split has too few rows. Check:
- `data_interim/coverage_report.json` — if coverage is <20%, the merge failed
- Your time range in `config.yaml → scope` — ensure it covers enough years

### Slow model training

The default config trains 200-300 trees per model × 5 models × 10 targets =
~15,000 fits. On a slow machine this can take 10+ minutes. Speed up:

```yaml
model:
  gradient_boosting:
    n_estimators: 100     # reduce from 300
    max_depth: 4          # reduce from 6
```

### Report shows "synthetic data" warning

The report template includes a caveat about synthetic data. When running on
real data, update the note in `src/report/generate_report.py` (search for
"synthetic data" in the `REPORT_TEMPLATE` string).

---

## 11. Production Hardening

### Parquet Output

After confirming `pyarrow` is installed:

```python
# In run_pipeline.py, change the save line:
featured.to_parquet(os.path.join(dirs["processed"], "master_panel.parquet"), index=False)
```

Parquet is ~5× smaller and ~10× faster to read than CSV.

### SHAP Integration

After confirming `shap` is installed, add to `src/models/train.py`:

```python
import shap

# After training the best model:
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val[:200])  # subsample for speed
shap.summary_plot(shap_values, X_val[:200], feature_names=feature_cols, 
                  show=False, max_display=15)
plt.savefig(f"{save_dir}/shap_{target}.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Quarto PDF Report

1. Install Quarto: https://quarto.org/docs/get-started/
2. Create `reports/final_report.qmd` wrapping the HTML content
3. Render: `quarto render reports/final_report.qmd --to pdf`

### CI/CD

The repo includes `.github/workflows/ci.yml` which runs:
- Linting with ruff
- Full synthetic pipeline test on Python 3.11 + 3.12
- Output artifact verification
- Weekly real-data test (triggered by `[test-real]` in commit message)

### Data Caching for CI

To avoid hitting APIs on every CI run, cache `data_raw/`:

```yaml
# In .github/workflows/ci.yml:
- uses: actions/cache@v4
  with:
    path: data_raw
    key: data-raw-${{ hashFiles('config.yaml') }}
```

### Logging

All pipeline output goes to both stdout and `pipeline.log`. For production,
redirect to a structured logger:

```bash
python run_pipeline.py --real 2>&1 | tee -a runs/$(date +%Y%m%d_%H%M%S).log
```

---

## 12. Extending the Pipeline

### Adding a New Dataset

1. Add entry to `registry/datasets.json` (source, license, fields, citation key)
2. Add BibTeX to `registry/citations.bib`
3. Write `ingest_<name>(raw_dir, cfg)` in `src/ingest/real_data.py`
   — must return a DataFrame with `country_iso3` and `year` columns
4. Register in `INGEST_REGISTRY` dict at the bottom of `real_data.py`
5. Add toggle to `config.yaml → data_sources.<name>: true`
6. Add endpoint config to `config.yaml → endpoints.<name>`
7. If new columns need special ranges, add to `VALID_RANGES` in `normalize.py`
8. If new columns need feature engineering, update `engineer.py`

### Adding a New Intervention

Edit `INTERVENTIONS` in `src/policy/planner.py`:

```python
{
    "tech_id": "HYDROGEN_GREEN",
    "name": "Green Hydrogen Production",
    "family": "hydrogen",
    "maturity_level": "emerging",
    "lead_time_years": 5,
    "lifetime_years": 20,
    "adoption_metric": "hydrogen_production_proxy",
    "sdg_links": ["SDG7", "SDG9", "SDG13"],
    "prerequisites_hard": ["GRID_UPGRADE", "SOLAR_SCALE"],
    "prerequisites_soft": ["capability_index > 0.6", "gdp_pc_ppp > 20000"],
    "estimated_effects": {
        "co2_per_capita_t": -0.5,
        "energy_intensity_mj_usd": -0.3,
    },
    "cost_category": "high",
    "description": "Electrolyzer capacity for industrial decarbonization."
}
```

### Adding a New Case Study Country

```bash
python run_pipeline.py --real --countries POL DEU IND BRA NGA
```

Or permanently in `config.yaml`:

```yaml
scope:
  case_study_countries: ["POL", "DEU", "IND", "BRA", "NGA"]
```

### Switching to EU NUTS2

This requires subnational data sources (Eurostat) and is a Phase 2 extension.
The schema is prepared for it:

```yaml
scope:
  geographic_unit: "nuts2-year"
```

You would need to:
1. Add Eurostat API ingestion to `real_data.py`
2. Add `region_id` as a key alongside `country_iso3`
3. Update the feature engineering to handle the nested hierarchy
4. Install `geopandas` for choropleth maps

---

## Quick Reference

```bash
# Zero to running (copy-paste block)
git clone https://github.com/konradpab/sdg-skill-trees.git
cd sdg-skill-trees
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
python preflight.py
python run_pipeline.py --synthetic    # verify
python run_pipeline.py --real         # real data
open reports/final_report.html
```
