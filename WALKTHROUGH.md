# Step-by-Step Implementation Walkthrough

Every command is copy-pasteable. Every step tells you what you should see,
how to verify it worked, and what to do if it didn't.

Estimated total time: 30–45 minutes (including data downloads).

---

## Overview: What We're Doing

```
                         ┌─────────────┐
                         │   GitHub     │
                         │   Repo       │
                         └──────┬──────┘
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌────────┐ ┌──────────────┐
              │ Codespace│ │ Local  │ │ Cloud VM     │
              │ (browser)│ │ laptop │ │ (Hetzner/AWS)│
              └────┬─────┘ └───┬────┘ └──────┬───────┘
                   │           │             │
                   └───────────┼─────────────┘
                               ▼
                    python run_pipeline.py --real
                               │
                               ▼
                    reports/final_report.html
```

You push the code once to GitHub. From there you can run the pipeline
anywhere — in a Codespace directly from the browser, on your laptop,
or on a cheap cloud VM. GitHub Actions tests every push automatically.

---

## Phase 1: Push the Code to GitHub

### 1a. Create the Repository

Go to https://github.com/new and create a new repository:

- **Name:** `sdg-skills-tree`
- **Visibility:** Private (you can make it public later for thesis submission)
- **Do NOT** initialize with a README — we already have one

### 1b. Initialize and Push

If you downloaded the `thesis_deploy` folder from our conversation:

```bash
cd thesis_deploy
git init
git add -A
git commit -m "Initial commit: full SDG skill-tree pipeline"
git branch -M main
git remote add origin git@github.com:<YOUR-USERNAME>/sdg-skills-tree.git
git push -u origin main
```

Replace `<YOUR-USERNAME>` with your GitHub username (e.g. `konradpab`).

**If you get a permission error on push**, you need to set up SSH keys:
```bash
# Generate a key (press Enter through all prompts)
ssh-keygen -t ed25519

# Copy the public key
cat ~/.ssh/id_ed25519.pub
```
Go to https://github.com/settings/keys → "New SSH key" → paste → Save.
Then retry `git push`.

**Alternatively, use HTTPS:**
```bash
git remote set-url origin https://github.com/<YOUR-USERNAME>/sdg-skills-tree.git
git push -u origin main
# Enter your GitHub username and a Personal Access Token (not your password)
```

### 1c. Verify on GitHub

Go to `https://github.com/<YOUR-USERNAME>/sdg-skills-tree`. You should see:

```
sdg-skills-tree/
├── .github/workflows/ci.yml
├── src/
├── registry/
├── tests/
├── config.yaml
├── run_pipeline.py
├── Dockerfile
├── Makefile
├── README.md
├── WALKTHROUGH.md
├── DEPLOYMENT.md
└── ...
```

Within a minute or two, go to the **Actions** tab. You should see the CI
workflow running (triggered by the push). It runs the full pipeline on
synthetic data. Once it finishes with a green ✓, you know the code works.

---

## Phase 2: Choose Where to Run the Real Pipeline

You need internet access to download the 9 datasets. Pick one:

### Option A: GitHub Codespaces (Easiest — Zero Local Setup)

This runs a full Linux VM in your browser, directly from the repo.

1. Go to your repo on GitHub
2. Click the green **"<> Code"** button → **"Codespaces"** tab
3. Click **"Create codespace on main"**
4. Wait ~2 minutes for the environment to build

You're now in VS Code in the browser with a terminal. Skip to **Phase 3**.

**Free tier:** 120 core-hours/month. A single pipeline run takes ~15 min
on a 4-core instance, so you have plenty of room.

### Option B: Your Local Machine

Any laptop/desktop with Python 3.10+ and internet access.

**System prerequisites:**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git curl

# macOS
brew install python@3.12 git

# Windows: install WSL2 first (wsl --install in PowerShell), then Ubuntu instructions
```

Clone your repo:
```bash
git clone git@github.com:<YOUR-USERNAME>/sdg-skills-tree.git
cd sdg-skills-tree
```

### Option C: Cloud VM (Best for Final Reproducible Runs)

A small cloud VM guarantees clean environment and fast downloads.

| Provider | Instance | Cost | Notes |
|----------|----------|------|-------|
| Hetzner | CX22 (2 vCPU, 4 GB) | €4.35/mo | Cheapest. EU datacenter. |
| GCP | e2-standard-2 (2 vCPU, 8 GB) | ~$0.07/hr | $300 free credits for new accounts |
| AWS | t3.large (2 vCPU, 8 GB) | ~$0.08/hr | |

Create an Ubuntu 24.04 server, SSH in, then:

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip git
git clone git@github.com:<YOUR-USERNAME>/sdg-skills-tree.git
cd sdg-skills-tree
```

---

## Phase 3: Set Up the Python Environment

This is the same regardless of where you're running.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. **Every command from here on
assumes the venv is active.** If you close the terminal and come back,
re-activate with `source .venv/bin/activate`.

```bash
pip install --upgrade pip
pip install -e .
```

Takes 2–5 minutes. If `lightgbm` fails to compile:
```bash
sudo apt install -y gcc g++ cmake libgomp1
pip install -e .
```

**Verify everything installed:**
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, shap, pyarrow, requests; print('All packages OK')"
```

---

## Phase 4: Pre-flight Check

```bash
python preflight.py
```

This validates your environment, project structure, and tests connectivity
to all 7 data source endpoints.

**What you want to see:**

```
  ✓ Python 3.12.x
  ✓ pandas, numpy, scikit-learn, xgboost, ... (all 15 green)
  ✓ Project structure — all files present
  ✓ Configuration — all sections valid
  ⚠ V-Dem data not found (optional — handled in Phase 5)
  ✓ Disk space
  ✓ World Bank API — 200
  ✓ UNSD SDG API — 200
  ✓ Ember CSV — 200
  ✓ Zenodo (PRIMAP) — 200
  ✓ SWIID — 200
  ✓ Carbon Pricing — 200
  ✓ UCDP GED — 200

  READY (or ALL CHECKS PASSED)
```

**If a package shows ✗:** Run `python preflight.py --fix` to auto-install.

**If an endpoint shows ✗:** That data source is down or your network
blocks it. Disable it in `config.yaml` (see Phase 5a) and re-check.

---

## Phase 5: Prepare Data Sources

### 5a. Review config.yaml

```bash
nano config.yaml    # or: code config.yaml (in Codespaces)
```

The `data_sources` section controls which datasets get downloaded:

```yaml
data_sources:
  unsd_sdg:       true
  ember:          true
  primap:         true
  wgi:            true
  vdem:           true    # ← set false if skipping the manual download
  wdi:            true
  swiid:          true
  carbon_pricing: true
  ucdp:           true
```

Set any source to `false` if its endpoint is down. The pipeline runs with
whatever is available.

### 5b. V-Dem Manual Download (Optional but Recommended)

V-Dem requires free registration. Without it you lose 3 democracy indices
but keep 6 governance indicators from WGI — still enough for the pipeline.

1. Go to https://v-dem.net/data/the-v-dem-dataset/
2. Click "Download" → register (free, instant)
3. Download **"Country-Year: V-Dem Core"** (CSV, ~200 MB)
4. Place it in the project:

```bash
mkdir -p data_raw/vdem
mv ~/Downloads/V-Dem-CY-Core-v14.csv data_raw/vdem/
ls -lh data_raw/vdem/    # verify: should show ~200 MB
```

**In Codespaces:** drag-and-drop the file into the file explorer sidebar,
then move it from the root into `data_raw/vdem/`.

**If you skip V-Dem**, set `vdem: false` in `config.yaml`.

### 5c. Check Fragile URLs

Ember and PRIMAP update their download URLs annually. Verify they're still live:

```bash
python test_source.py ember primap
```

You should see `✓ OK` for both. If either shows `✗ FAILED`:

- **Ember 404:** Go to https://ember-energy.org/data/yearly-electricity-data/,
  click Download, copy the new CSV URL, and update
  `config.yaml → endpoints.ember.yearly_url`

- **PRIMAP 404:** Go to https://zenodo.org/search?q=primap-hist, find the
  latest record, copy the CSV link, update
  `config.yaml → endpoints.primap.download_url`

You can test all sources at once with `python test_source.py --all`.

---

## Phase 6: Smoke Test with Synthetic Data

Before hitting real APIs, verify the full pipeline:

```bash
python run_pipeline.py --synthetic
```

Takes ~3 minutes. You should see all 7 stages complete and end with:

```
PIPELINE COMPLETE — Time: ~180s
```

Verify:
```bash
python verify_run.py
```

All green ✓ means the code works. Proceed to real data.

---

## Phase 7: Run on Real Data

```bash
python run_pipeline.py --real 2>&1 | tee run_real.log
```

**Expected timeline:**

| Stage | Duration | What's happening |
|-------|----------|-----------------|
| Ingest | 3–8 min | Downloading 9 datasets from APIs |
| Normalize | <10 sec | Clipping, interpolating, coverage report |
| Features | <30 sec | Building 160+ feature columns |
| Training | 2–5 min | 50 models (5 families × 10 targets) |
| Recommend | <10 sec | Scoring 13 interventions for 3 countries |
| Visualize | <30 sec | 9 PNG figures |
| Report | <10 sec | HTML with embedded images |
| **Total** | **5–15 min** | |

### Watch Ingestion

The log shows each dataset being fetched:

```
[UNSD] Fetching elec_access (EG_ELC_ACCS)...
  elec_access: 1847 rows
[EMBER] Downloading from https://ember-energy.org/...
[OK] ember: 18432 rows
[WGI] Fetching governance indicators...
[PRIMAP] Downloading from Zenodo...
[OK] primap: 34521 rows
```

**If a source fails**, you'll see `[FAIL] source_name: ConnectionError`.
That's OK — the pipeline continues with whatever succeeded. Check what
made it through:

```bash
python -c "import json; print(json.load(open('data_raw/ingestion_manifest.json'))['sources_ingested'])"
```

### After Completion

```bash
python verify_run.py
```

Key checks:
- `master_panel.csv` should be >1 MB
- Panel should cover ≥100 countries (vs 50 in synthetic)
- Models beat naive baseline on most targets
- All 3 roadmap CSVs exist

Inspect data quality:
```bash
python -c "
import json
r = json.load(open('data_interim/coverage_report.json'))
print(f'Countries: {r[\"n_countries\"]}, Years: {r[\"n_years\"]}, Rows: {r[\"n_rows\"]}')
for col, info in sorted(r['column_coverage'].items()):
    pct = info['coverage_pct']
    print(f'  {\"✓\" if pct > 60 else \"⚠\" if pct > 30 else \"✗\"} {col}: {pct}%')
"
```

### View the Report

```bash
# In Codespaces:
#   Right-click reports/final_report.html → "Open with Live Server"
#   Or: click the file, then the preview icon in the top right

# On local machine:
open reports/final_report.html          # macOS
xdg-open reports/final_report.html      # Linux

# On a cloud VM — download to your laptop:
scp root@<server-ip>:~/sdg-skills-tree/reports/final_report.html .
open final_report.html
```

The report auto-detects real vs synthetic data. With real data (≥100 countries)
it replaces the synthetic-data caveat with a proper observational-data disclaimer.

---

## Phase 8: Commit Results to GitHub

The source code is already on GitHub. Now decide how to handle the outputs
(which are gitignored because they're large).

### Option A: GitHub Releases (Recommended for Thesis)

Package the outputs as a release artifact that anyone can download:

```bash
tar czf thesis_outputs_v1.0.tar.gz \
  data_processed/master_panel.csv \
  models_saved/training_results.json \
  models_saved/feature_importances.json \
  outputs/recommendations/ \
  reports/final_report.html \
  reports/figures/ \
  data_interim/coverage_report.json \
  pipeline.log
```

Then on GitHub:
1. Go to your repo → **Releases** → **"Create a new release"**
2. Tag: `v1.0`
3. Title: `Thesis pipeline run — real data`
4. Description: paste the output of `python verify_run.py`
5. Attach `thesis_outputs_v1.0.tar.gz` and `reports/final_report.html`
6. Publish

Now anyone (including your thesis committee) can download the report directly
from the Releases page without running the pipeline.

### Option B: Git LFS (If You Want Outputs Versioned in the Repo)

```bash
git lfs install
git lfs track "data_processed/*.csv"
git lfs track "data_processed/*.parquet"
git lfs track "models_saved/*.pkl"
git lfs track "reports/final_report.html"
git add .gitattributes
git commit -m "Track large files with LFS"
```

Then remove the relevant lines from `.gitignore`, add the files, and push.
GitHub gives you 1 GB of free LFS storage.

### Option C: Just Commit Config Changes

If you changed `config.yaml`, utility weights, or fixed an endpoint URL,
commit those so the next run picks them up:

```bash
git add config.yaml
git commit -m "Update Ember URL for 2025 release, tune CO2 weight"
git push
```

---

## Phase 9: Fine-Tune and Iterate

### Change Case Study Countries

```yaml
# config.yaml
scope:
  case_study_countries: ["POL", "DEU", "BRA"]
```

```bash
python run_pipeline.py --from-cache --skip-models
# Re-generates recommendations and report without re-downloading or re-training
```

Or on the command line without editing config:
```bash
python run_pipeline.py --from-cache --skip-models --countries POL NLD ZAF
```

### Adjust Utility Weights

If Poland's roadmap doesn't emphasize coal exit enough:

```yaml
# config.yaml
planner:
  weights:
    w_co2: 0.40          # up from 0.30
    w_feasibility: 0.10   # down from 0.15
```

```bash
python run_pipeline.py --from-cache --skip-models
```

### Add 5-Year Forecasts

In `src/features/engineer.py`, change the `add_forward_targets` call:

```python
df = add_forward_targets(df, TARGET_COLS, horizons=(1, 3, 5))
```

In `src/models/train.py`, add entries to `TARGET_NAMES`:

```python
"target_renewables_share_pct_delta5y": "Δ Renewables Share (5y)",
"target_co2_per_capita_t_delta5y": "Δ CO₂ per capita (5y)",
```

Then retrain (features changed, so skip only ingestion):
```bash
python run_pipeline.py --from-cache
```

### Enable SHAP Plots

Add to `src/viz/plots.py`:

```python
import shap

def plot_shap_summary(model, X, feature_names, save_path, max_display=15):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:300])
    shap.summary_plot(shap_values, X[:300], feature_names=feature_names,
                      show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

### Switch to Parquet Output

In `run_pipeline.py`, change the save line (~line 117):

```python
featured.to_parquet(os.path.join(dirs["processed"], "master_panel.parquet"), index=False)
```

5× smaller, 10× faster to read.

### Push Your Changes

After any tuning:

```bash
git add -A
git commit -m "Tune weights, add 5y forecasts, enable SHAP"
git push
```

CI runs automatically on push and verifies the pipeline still works.

---

## Phase 10: CI / Continuous Integration

### What Runs Automatically

Every push to `main` triggers `.github/workflows/ci.yml`:

1. Install dependencies (Python 3.11 + 3.12)
2. Lint with ruff
3. Run pytest on synthetic data
4. Run the full synthetic pipeline
5. Verify all output files exist
6. Upload `reports/final_report.html` as a downloadable artifact

Go to the **Actions** tab on GitHub to see results.

### Trigger a Real-Data Test

The CI also has a real-data job that runs when you include `[test-real]`
in a commit message:

```bash
git commit --allow-empty -m "Test real data endpoints [test-real]"
git push
```

This runs the full pipeline against live APIs and uploads the report +
coverage report + logs as artifacts. Check the Actions tab for results.

### Download CI Artifacts

After any CI run completes:
1. Go to **Actions** → click the run
2. Scroll to **Artifacts** at the bottom
3. Download `report-py3.12` (or `report-real-data`)

This lets you review the report without running anything locally.

---

## Phase 11: Docker (Thesis Submission Reproducibility)

For your thesis appendix: "Anyone can reproduce these results with two commands."

### Build and Run Locally

```bash
docker build -t sdg-skills-tree .

# Synthetic test:
docker run sdg-skills-tree --synthetic

# Real data with output persistence:
docker run \
  -v $(pwd)/reports:/thesis/reports \
  -v $(pwd)/outputs:/thesis/outputs \
  -v $(pwd)/data_processed:/thesis/data_processed \
  sdg-skills-tree --real
```

### Or Use Docker Compose

```bash
docker compose up pipeline          # real data, outputs mounted
docker compose run pipeline --synthetic   # test mode
docker compose run --rm test              # pytest
```

### Push to GitHub Container Registry

So anyone can pull your image without building:

```bash
# Authenticate to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u <YOUR-USERNAME> --password-stdin

# Tag and push
docker tag sdg-skills-tree ghcr.io/<YOUR-USERNAME>/sdg-skills-tree:v1.0
docker push ghcr.io/<YOUR-USERNAME>/sdg-skills-tree:v1.0
```

Then anyone can run:
```bash
docker run ghcr.io/<YOUR-USERNAME>/sdg-skills-tree:v1.0 --real
```

---

## Troubleshooting

### "World Bank API returns [null, null]"

API is overloaded. Retry logic handles transient failures, but if it persists:

```bash
curl "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.KD?date=2020&format=json&per_page=5"
```

If genuinely down, set `wgi: false` and `wdi: false` in `config.yaml` and re-run.

### "Ember CSV has unexpected columns"

Ember restructures their CSV annually. Check the downloaded file:

```bash
head -5 data_raw/ember/yearly_full_release.csv
```

Compare with what `ingest_ember()` expects in `src/ingest/real_data.py`.

### "PRIMAP download returns a zip instead of CSV"

Newer releases are sometimes zipped. Update `ingest_primap()` in `real_data.py`:

```python
if resp.content[:2] == b'PK':  # ZIP magic bytes
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))
        df = pd.read_csv(zf.open(csv_name))
else:
    df = pd.read_csv(io.BytesIO(resp.content))
```

### "Panel has only 30 countries"

Country name mapping is failing. Check the log:

```bash
grep "unmapped\|cannot find\|None" pipeline.log
```

Add missing name variants to `_NAME_TO_ISO3` in `src/ingest/country_codes.py`.

### "All models perform same as naive baseline"

Features are mostly NaN — the merge failed. Diagnose:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data_processed/master_panel.csv')
print('Shape:', df.shape)
miss = df.isnull().mean().sort_values(ascending=False)
print(miss.head(20))
"
```

If >90% null, the `country_iso3 × year` join isn't working. Most common cause:
one dataset uses ISO2 codes while others use ISO3. Check `pipeline.log`.

### "Report images are blank/missing"

A column referenced in `plots.py` doesn't exist in real data:

```bash
grep "Error\|Traceback" pipeline.log
python -c "import pandas as pd; print(sorted(pd.read_csv('data_processed/master_panel.csv').columns.tolist()))"
```

---

## Quick Command Reference

```bash
# ── Setup ──────────────────────────────────────────────
git clone git@github.com:<YOUR-USERNAME>/sdg-skills-tree.git && cd sdg-skills-tree
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
python preflight.py

# ── Run ────────────────────────────────────────────────
python run_pipeline.py --synthetic                 # smoke test
python run_pipeline.py --real                      # full real run
python run_pipeline.py --from-cache                # skip downloads
python run_pipeline.py --from-cache --skip-models  # fastest iteration

# ── Verify ─────────────────────────────────────────────
python verify_run.py
python test_source.py --all                        # test endpoints

# ── Iterate ────────────────────────────────────────────
make run-cached                  # re-run from cache
make run-fast                    # cache + skip models
make clean                       # clear outputs only
make test                        # pytest

# ── Git ────────────────────────────────────────────────
git add -A && git commit -m "..." && git push      # CI runs automatically
git commit --allow-empty -m "[test-real]" && git push  # trigger real-data CI

# ── Docker ─────────────────────────────────────────────
docker compose up pipeline                         # real data
docker compose run pipeline --synthetic            # test mode
```
