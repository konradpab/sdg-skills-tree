# SDG Skill-Trees: ML Pipeline for Country-Specific Decarbonization Roadmaps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23649099.svg)](https://doi.org/10.5281/zenodo.23649099)

Code for the MSc thesis **"From Indicators to Interventions: Learning Personalized
Policy–Tech Pathways Toward SDG Progress"** (Konrad Pabianczyk, OPIT – Open Institute
of Technology, 2026). The thesis is open access: https://doi.org/10.5281/zenodo.23649099

A reproducible ML system that learns region-specific technology/policy adoption
pathways ("skill-trees") to maximize SDG progress and decarbonization while
respecting feasibility constraints, and produces actionable ranked roadmaps
with uncertainty quantification.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/konradpab/sdg-skills-tree && cd sdg-skills-tree

# 2. Create virtual environment + install
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. (Optional) Place V-Dem CSV manually — see "Manual Downloads" below
#    (Optional) Place the Hofstede CSV for the culture features — see DATA_SOURCES.md §10

# 4. Run the full pipeline on real data
python run_pipeline.py --real

# Or test with synthetic data first (no network needed)
python run_pipeline.py --synthetic
```

The pipeline takes 3–10 minutes on real data (depending on API response times)
and produces `reports/final_report.html` plus all intermediate artifacts.

---

## Requirements

### System

| Requirement     | Minimum          | Recommended        |
|-----------------|------------------|--------------------|
| Python          | 3.10             | 3.11 or 3.12       |
| RAM             | 4 GB             | 8 GB               |
| Disk            | 2 GB free        | 5 GB free           |
| Network         | Required for `--real` mode | Broadband      |
| OS              | Linux, macOS, or WSL2 on Windows | Ubuntu 22.04+ |

### Python Packages

All dependencies are declared in `pyproject.toml`. The core install (`pip install -e .`)
gives you everything needed for the full pipeline. Optional extras:

```bash
pip install -e ".[dev]"          # + pytest, ruff, pre-commit
pip install -e ".[quarto]"       # + Jupyter (for Quarto PDF rendering)
pip install -e ".[extensions]"   # + geopandas, plotly, causal-learn
pip install -e ".[dev,quarto,extensions]"  # everything
```

**Key packages and why they're needed:**

| Package        | Purpose                                   |
|----------------|-------------------------------------------|
| `pandas`       | Panel data manipulation                   |
| `pyarrow`      | Parquet I/O (faster than CSV)             |
| `openpyxl`     | Reading Excel files (Carbon Pricing, WGI) |
| `scikit-learn` | Baselines, random forest, gradient boosting |
| `xgboost`      | XGBoost gradient boosting                 |
| `lightgbm`     | LightGBM gradient boosting                |
| `shap`         | SHAP feature importance                   |
| `networkx`     | Intervention DAG                          |
| `requests`     | HTTP downloads / API calls                |
| `pyyaml`       | Config file parsing                       |
| `jinja2`       | HTML report templating                    |

### Optional: Quarto

For a publication-quality PDF report, install [Quarto](https://quarto.org/docs/get-started/).
The pipeline generates a self-contained HTML report by default; Quarto is only
needed if you want to render `reports/final_report.qmd` to PDF.

```bash
# macOS
brew install quarto

# Ubuntu/Debian
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.554/quarto-1.4.554-linux-amd64.deb
sudo dpkg -i quarto-*.deb

# Verify
quarto check
```

---

## Project Structure

```
sdg-skills-tree/
├── config.yaml                  # All tuneable parameters (single source of truth)
├── run_pipeline.py              # Main entry point
├── pyproject.toml               # Dependencies
├── Makefile                     # Convenience commands
├── .env.example                 # Environment variable template
├── .gitignore
│
├── src/
│   ├── ingest/
│   │   ├── real_data.py         # Real API/download ingestion (9 datasets)
│   │   ├── synthetic_data.py    # Synthetic data for testing
│   │   └── country_codes.py     # ISO3 mapping utilities
│   ├── clean/
│   │   └── normalize.py         # Range checks, interpolation, coverage report
│   ├── features/
│   │   └── engineer.py          # Lags, deltas, rolling, composite indices
│   ├── models/
│   │   └── train.py             # Baselines, GBR, quantile, conformal
│   ├── policy/
│   │   └── planner.py           # Intervention graph + recommendation engine
│   ├── viz/
│   │   └── plots.py             # All report figures
│   └── report/
│       └── generate_report.py   # HTML report via Jinja2
│
├── registry/
│   ├── datasets.json            # Machine-readable dataset registry
│   └── citations.bib            # BibTeX for all sources
│
├── data_raw/                    # Downloaded raw files (gitignored)
├── data_interim/                # Coverage reports, QC outputs
├── data_processed/              # master_panel.csv / .parquet
├── models_saved/                # Pickled models, metrics JSON
├── outputs/recommendations/     # Per-country roadmap CSVs
└── reports/
    ├── figures/                 # Generated PNGs
    └── final_report.html        # Self-contained HTML report
```

---

## Running the Pipeline

### Mode 1: Real Data (default)

```bash
python run_pipeline.py --real
```

This downloads all 9 MVS datasets from their public endpoints, normalizes them,
trains models, generates recommendations, and produces the report. Takes 3–10 min.

### Mode 2: Synthetic Data (testing / no network)

```bash
python run_pipeline.py --synthetic
```

Uses the built-in synthetic data generator. Identical pipeline code, ~3 min,
no network required. Use this to verify your environment works before switching
to real data.

### Mode 3: From Cache (re-run without re-downloading)

```bash
python run_pipeline.py --from-cache
```

After a successful `--real` run, uses the saved `data_raw/master_panel_raw.csv`
to skip the download step. Useful for iterating on features/models.

### Additional Flags

```bash
--skip-models          # Skip training, use models_saved/ from a previous run
--countries POL DEU BRA  # Override case study countries
```

### Makefile Shortcuts

```bash
make install           # Install dependencies
make run               # Full pipeline (real data)
make run-synthetic     # Synthetic mode
make run-cached        # From cache
make run-fast          # Cache + skip models (fastest iteration)
make clean             # Remove generated outputs (keep raw data)
make clean-all         # Remove everything including downloads
make test              # Run pytest
make lint              # Lint with ruff
make vdem-check        # Check if V-Dem data is present
```

---

## Data Sources: What's Automatic vs. Manual

### Fully Automatic (API / direct download)

These datasets are fetched by the pipeline with no user action:

| Dataset            | Endpoint                              | Auth | Notes |
|--------------------|---------------------------------------|------|-------|
| UNSD SDG           | REST API (unstats.un.org)             | None | ~6 series, paginated JSON |
| World Bank WDI     | REST API v2 (api.worldbank.org)       | None | ~7 indicators |
| World Bank WGI     | REST API v2 (api.worldbank.org)       | None | 6 governance indicators |
| SWIID              | Direct zip download (fsolt.org)       | None | Gini indices |
| UCDP GED           | Direct zip download (ucdp.uu.se)      | None | Conflict events, aggregated |
| Carbon Pricing     | Direct xlsx (carbonpricingdashboard)  | None | Price + coverage |
| Ember Electricity  | Direct CSV (ember-energy.org)         | None | Generation mix, emissions |
| PRIMAP-hist        | Zenodo CSV download                   | None | Historical emissions |

### Manual Download Required

| Dataset | Why Manual | Instructions |
|---------|-----------|--------------|
| **V-Dem** | Requires free registration + license agreement | See below |

#### V-Dem Download Steps

1. Go to https://v-dem.net/data/the-v-dem-dataset/
2. Click "Download" → register for a free account
3. Download **"Country-Year: V-Dem Core"** (CSV format)
4. Place the CSV file in `data_raw/vdem/`
5. Run `make vdem-check` to verify

If you skip V-Dem, the pipeline runs without it — you just lose the democracy
indices (`v2x_polyarchy`, `v2x_libdem`, `v2x_partipdem`).

---

## Configuration

All tuneable parameters live in `config.yaml`. Key sections:

### Scope

```yaml
scope:
  train_start: 1996
  train_end: 2023
  forecast_horizons: [1, 3]
  case_study_countries: ["POL", "DEU", "IND"]
```

### Data Source Toggles

Disable any source that gives you trouble:

```yaml
data_sources:
  unsd_sdg:       true
  ember:          true
  primap:         true
  wgi:            true
  vdem:           true    # set false if you don't have the manual download
  wdi:            true
  swiid:          true
  carbon_pricing: true
  ucdp:           true
```

### Model Configuration

```yaml
model:
  train_cutoff: 2017       # train: ≤2017
  val_cutoff: 2020         # val: 2018-2020, test: 2021+
  gradient_boosting:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
```

### Planner Utility Weights

```yaml
planner:
  weights:
    w_sdg:         0.25
    w_co2:         0.30
    w_equity:      0.10
    w_feasibility: 0.15
    w_speed:       0.10
    w_cost:        0.10
```

---

## Troubleshooting

### "Connection refused" / download failures

Some institutional networks block outbound HTTP to non-standard APIs.

**Fix:** Try from a personal network or a cloud VM. The World Bank API and
Zenodo are generally not blocked; Ember and UCDP occasionally have downtime.

After any successful download, future runs can use `--from-cache`.

### "No CSV found in zip" (SWIID or UCDP)

The download URL may have changed. Check:
- SWIID: https://fsolt.org/swiid/ for the current zip link
- UCDP: https://ucdp.uu.se/downloads/ for the current GED version

Update the URL in `config.yaml` under `endpoints.swiid.download_url` or
`endpoints.ucdp.ged_url`.

### Ember CSV format changed

Ember occasionally restructures their CSV. If parsing fails:
1. Download manually from https://ember-energy.org/data/yearly-electricity-data/
2. Place in `data_raw/ember/yearly_full_release.csv`
3. Check column names against `src/ingest/real_data.py:ingest_ember()`

### PRIMAP Zenodo URL returns 404

New PRIMAP releases get a new Zenodo record ID. Go to
https://zenodo.org/search?q=primap-hist and find the latest record.
Update `config.yaml` → `endpoints.primap.download_url`.

### World Bank API returns empty or `[null, null]`

The WB API occasionally returns empty for recent years. This is normal —
the pipeline handles missing values via the normalization layer.
If an entire indicator is empty, check if the indicator code changed at
https://data.worldbank.org/indicator.

### UNSD SDG API returns unexpected structure

The UNSD API has been restructured several times. If parsing fails:
1. Visit https://unstats.un.org/sdgapi/swagger/ to check the current schema
2. Test a manual query: `curl "https://unstats.un.org/sdgapi/v1/sdg/Series/Data?seriesCode=EG_ELC_ACCS&pageSize=5"`
3. Update `src/ingest/real_data.py:ingest_unsd_sdg()` to match the new response structure

### Pipeline runs but report shows empty tables

Check `data_interim/coverage_report.json` — if most columns have <20% coverage,
the merge likely failed due to mismatched country codes. Look at `pipeline.log`
for warnings about unmapped countries.

---

## Recommended Environment

### Option A: Local Machine (simplest)

Any modern laptop/desktop with Python 3.10+ and internet access.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python run_pipeline.py --synthetic   # verify
python run_pipeline.py --real        # real data
```

### Option B: Cloud VM (recommended for reproducibility)

A small cloud VM guarantees a clean environment and reliable network.

**Google Cloud / AWS / Azure:**
- Instance: `e2-standard-2` (2 vCPU, 8 GB RAM) or equivalent — ~$0.07/hr
- OS: Ubuntu 22.04 LTS
- Disk: 20 GB SSD
- Runtime: ~10 min for full pipeline

```bash
# On a fresh Ubuntu 22.04 VM:
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git
git clone https://github.com/konradpab/sdg-skills-tree && cd sdg-skills-tree
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
python run_pipeline.py --real
# Download report: scp user@vm:sdg-skills-tree/reports/final_report.html .
```

### Option C: GitHub Codespaces (zero setup)

1. Push this repo to GitHub
2. Open in Codespaces (free tier: 120 core-hours/month)
3. Terminal: `pip install -e . && python run_pipeline.py --real`

### Option D: Docker (maximum reproducibility)

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /thesis
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "run_pipeline.py", "--real"]
```

```bash
docker build -t sdg-skills-tree .
docker run -v $(pwd)/outputs:/thesis/outputs sdg-skills-tree
```

---

## Extending the Pipeline

### Adding a new dataset

1. Add the entry to `registry/datasets.json`
2. Add the BibTeX to `registry/citations.bib`
3. Write an `ingest_<name>()` function in `src/ingest/real_data.py`
   - Must return a DataFrame with `country_iso3` and `year` columns
4. Register it in `INGEST_REGISTRY` at the bottom of that file
5. Add a toggle in `config.yaml` → `data_sources`
6. If the new columns need special feature engineering, update `src/features/engineer.py`

### Adding a new intervention node

Edit `INTERVENTIONS` in `src/policy/planner.py`. Each node needs:
- `tech_id`, `name`, `family`
- `prerequisites_hard` (list of tech_ids that must be adopted first)
- `prerequisites_soft` (list of conditions like `"gov_effectiveness > 0.4"`)
- `estimated_effects` (dict of indicator deltas)
- `lead_time_years`, `lifetime_years`, `cost_category`, `sdg_links`

### Switching to Parquet output

Once `pyarrow` is installed:
```python
# In run_pipeline.py, replace:
featured.to_csv(..., "master_panel.csv", index=False)
# With:
featured.to_parquet(..., "master_panel.parquet", index=False)
```

### Adding SHAP plots

After `pip install shap`:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
plt.savefig("reports/figures/shap_summary.png")
```

### Upgrading to Quarto PDF

1. Install Quarto (see Requirements section above)
2. Convert the HTML template in `src/report/generate_report.py` to a `.qmd` file
3. Run: `quarto render reports/final_report.qmd --to pdf`

---

## Output Artifacts

After a successful run:

| Path | Contents |
|------|----------|
| `data_processed/master_panel.csv` | Full feature matrix (countries × years × features) |
| `data_processed/dim_country.csv` | Country dimension table |
| `models_saved/training_results.json` | All model metrics (train/val/test) |
| `models_saved/feature_importances.json` | Top features per target |
| `models_saved/models.pkl` | Pickled best models per target |
| `models_saved/quantile_models.pkl` | Quantile regression models |
| `models_saved/conformal_predictors.pkl` | Calibrated conformal predictors |
| `outputs/recommendations/POL_roadmap.csv` | Poland intervention roadmap |
| `outputs/recommendations/DEU_roadmap.csv` | Germany intervention roadmap |
| `outputs/recommendations/IND_roadmap.csv` | India intervention roadmap |
| `outputs/recommendations/all_recommendations.json` | Full recommendation objects |
| `reports/final_report.html` | Self-contained HTML report with embedded figures |
| `reports/figures/*.png` | Individual figures |
| `registry/datasets.json` | Dataset registry with provenance |
| `registry/citations.bib` | BibTeX citations |
| `pipeline.log` | Full execution log |

---

## Citation

If you use this pipeline or its findings, please cite the thesis
(GitHub's "Cite this repository" button reads `CITATION.cff`):

```bibtex
@mastersthesis{pabianczyk_2026_indicators,
  author = {Pabianczyk, Konrad},
  title  = {From Indicators to Interventions: Learning Personalized Policy--Tech Pathways Toward SDG Progress},
  school = {OPIT -- Open Institute of Technology},
  year   = {2026},
  doi    = {10.5281/zenodo.23649099},
  url    = {https://doi.org/10.5281/zenodo.23649099}
}

@software{pabianczyk_2026_sdg_skill_trees,
  author = {Pabianczyk, Konrad},
  title  = {SDG Skill-Trees: ML Pipeline for Country-Specific Decarbonization Roadmaps},
  year   = {2026},
  url    = {https://github.com/konradpab/sdg-skills-tree}
}
```

---

## Acknowledgements

Developed as the MSc thesis in Applied AI and Data Science at OPIT – Open Institute
of Technology (2026).

## License

Code: MIT (see `LICENSE`). Data sources keep their own licences; see
`DATA_SOURCES.md`. Hofstede dimension scores are not redistributed here.
