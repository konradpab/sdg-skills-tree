# Final Implementation Guide

Everything you need to go from zero to a complete pipeline run.

---

## What's In This Package

**Models:**
- 6 model families per target: naive persistence, elastic net, random forest, gradient boosting, histogram GB, **stacking ensemble** (new)
- Ensemble combines 4 base learners via Ridge meta-learner — auto-selected when it outperforms individuals
- Forecasts at **1-year, 3-year, and 5-year** horizons

**Backtesting (new chapter):**
- **Policy alignment**: did indicators move in the direction we recommended?
- **Forecast accuracy**: predicted vs actual deltas, directional accuracy per country
- **Recommendation stability**: Jaccard similarity across base years 2015-2017
- 3 new figures auto-generated for the report

**Data sources — 9 core, 5 extensions:**
- 5 auto-download via API (WDI, WGI, SWIID, Carbon Pricing, UCDP)
- 2 auto-download but URLs may need updating (UNSD SDG, PRIMAP)
- 2 require manual download (Ember, V-Dem)
- 5 optional extensions documented (UNESCO, WVS, GDELT, OEC, WID)

---

## Step 1: Delete Old Repo, Create Fresh One

1. Go to `https://github.com/YOUR_USERNAME/sdg-skills-tree`
2. **Settings** → scroll to bottom → **Delete this repository** → confirm
3. Go to https://github.com/new
4. Name: `sdg-skills-tree`, Private ✓, do NOT check any boxes → **Create repository**

## Step 2: Upload Files

1. Download `sdg-skills-tree-final.tar.gz` from this conversation
2. Unzip it on your computer
3. Inside you'll find a folder called `sdg-final` — open it
4. On the empty GitHub repo page, click **"uploading an existing file"**
5. Select **everything inside `sdg-final`** (Ctrl+A / Cmd+A) and drag into the upload area
6. Click **Commit changes**

## Step 3: Verify Upload

On the repo page you should see these at the top level:

```
setup.sh          ← this is critical
config.yaml
run_pipeline.py
pyproject.toml
src/              ← click in: should contain ingest/, clean/, features/, models/, policy/, viz/, report/
registry/         ← click in: should contain datasets.json, citations.bib
.devcontainer/    ← may be hidden (see Step 4)
```

Click into `src/policy/` — you should see **backtest.py** alongside planner.py. That confirms the new backtesting module uploaded correctly.

## Step 4: Add Hidden Files (if GitHub didn't upload them)

GitHub's upload sometimes skips files starting with `.` Check if these exist:

**Check `.devcontainer/devcontainer.json`:** Click through the repo file listing. If you don't see a `.devcontainer` folder:

1. Click **Add file → Create new file**
2. Type `.devcontainer/devcontainer.json` as the filename
3. Paste:
```json
{
  "name": "SDG Skill-Trees Pipeline",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "postCreateCommand": "bash setup.sh",
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "terminal.integrated.defaultProfile.linux": "bash"
      },
      "extensions": ["ms-python.python"]
    }
  }
}
```
4. Commit

**Check `.gitignore`:** If missing, same process — create `.gitignore` and paste:
```
data_raw/
data_interim/
data_processed/
!data_raw/.gitkeep
!data_interim/.gitkeep
!data_processed/.gitkeep
models_saved/
!models_saved/.gitkeep
outputs/recommendations/*.csv
outputs/recommendations/*.json
reports/figures/
reports/final_report.html
!reports/.gitkeep
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
dist/
build/
.env
*.log
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
```

## Step 5: Create Codespace

1. Green **`<> Code`** button → **Codespaces** tab → **Create codespace on main**
2. Wait **3-5 minutes** — `setup.sh` runs automatically and installs all packages
3. At the end of the build log you should see: `✅ All 15 packages installed successfully`

**If packages didn't install** (you see red errors or the terminal shows no packages):
```
bash setup.sh
```
Wait 2-3 minutes. This always works regardless of what the devcontainer did.

## Step 6: Verify Environment

In the terminal (Ctrl+` if you don't see it):

```
python preflight.py --env-only
```

You want all 15 packages green ✓. If any show ✗:
```
python preflight.py --fix
```

## Step 7: Smoke Test

```
python run_pipeline.py --synthetic
```

Wait ~3-5 minutes. Should end with:
```
PIPELINE COMPLETE
  Time: ~250s
```

New things you'll see compared to before:
- `Ensemble stack val_mae=0.XXXX` during training (the ensemble competing)
- `STAGE 5b: Backtesting & Retrospective Analysis` with alignment scores
- Backtest figures in the output

Verify:
```
python verify_run.py
```

## Step 8: Download Ember Data (Manual — Required)

Ember blocks automated downloads. You must do this manually:

1. On your regular computer, go to: https://ember-energy.org/data/yearly-electricity-data/
2. Download the **"Yearly electricity data"** CSV (long format)
3. In the Codespace terminal:
   ```
   mkdir -p data_raw/ember
   ```
4. In the left sidebar, find `data_raw/ember/`
5. **Drag and drop** the downloaded CSV file into that folder
6. Rename it if needed:
   ```
   mv data_raw/ember/*.csv data_raw/ember/yearly_full_release_long_format.csv
   ```

## Step 9: Run on Real Data

```
python run_pipeline.py --real 2>&1 | tee run_real.log
```

Takes **5-15 minutes**. Watch for:

```
>>> STAGE 1: Data Ingestion
  [WDI] Fetching macro indicators...        ← should work
  [WGI] Fetching governance indicators...   ← should work
  [EMBER] Reading from local file...        ← uses your manual download
  [PRIMAP] Downloading from Zenodo...       ← updated URL
  ... 

>>> STAGE 4: Model Training
  Training for: Δ CO₂ per capita (1y)
    Ensemble stack val_mae=0.XXXX           ← ensemble competing
    Best model: ensemble_stack              ← ensemble wins when it's best

>>> STAGE 5b: Backtesting & Retrospective Analysis
  Policy alignment backtests:
    POL (3y): alignment=0.XX, weighted=0.XX
    DEU (3y): alignment=0.XX, weighted=0.XX
    IND (3y): alignment=0.XX, weighted=0.XX
  Recommendation stability:
    POL: mean Jaccard=0.XX, top-3 stable=True
  Forecast accuracy backtest:
    renewables_share_pct_delta3y: direction_accuracy=0.XX

>>> PIPELINE COMPLETE
```

## Step 10: Check Results

```
python verify_run.py
```

Also check the backtest output:
```
cat outputs/recommendations/backtest_results.json | python -m json.tool | head -40
```

## Step 11: View and Download Report

Right-click `reports/final_report.html` in the sidebar → **Download** → open in your browser.

The report now includes:
- **Model comparison** showing the ensemble alongside individual models
- **Backtest alignment** chart (did recommendations match reality?)
- **Forecast accuracy** chart (directional accuracy per indicator)
- **Stability** chart (do recommendations change when base year shifts?)
- **5-year forecast** results alongside 1-year and 3-year

Download everything:
```
tar czf results.tar.gz \
  reports/final_report.html reports/figures/ \
  outputs/recommendations/ \
  data_processed/master_panel.csv \
  models_saved/training_results.json \
  models_saved/feature_importances.json \
  data_interim/coverage_report.json \
  pipeline.log
```
Right-click `results.tar.gz` in sidebar → **Download**

## Step 12: Iterate

**Change countries:**
```
python run_pipeline.py --from-cache --skip-models --countries POL DEU BRA
```

**Change utility weights** — edit `config.yaml`, then:
```
python run_pipeline.py --from-cache --skip-models
```

**Retrain models** (after changing features or targets):
```
python run_pipeline.py --from-cache
```

**Save changes to GitHub:**
1. Click Source Control icon in sidebar (branch symbol)
2. Type a message → **Commit** → **Sync Changes**

## Step 13: Stop Codespace

Always stop when done to save free hours:
1. Bottom-left corner → click "Codespaces" → **Stop Current Codespace**
2. Or: https://github.com/codespaces → `...` → **Stop codespace**

Restart later: https://github.com/codespaces → click your codespace

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Terminal shows `$` but nothing else | Click inside the terminal, type commands there |
| No terminal visible | Press Ctrl+` |
| `bash setup.sh` fails on gcc | Run: `sudo apt-get update && sudo apt-get install -y gcc g++ cmake` then retry |
| Ember shows 403 | Manual download required (Step 8) |
| PRIMAP returns empty | URL may have changed — check https://zenodo.org/search?q=primap-hist |
| WGI shows EMPTY in test | Normal for small samples — full pipeline fetches all years and works |
| Only 50 countries in results | You ran `--synthetic` not `--real` |
| Codespace build failed | Delete at github.com/codespaces, recreate |
| Pipeline takes >20 min | Some APIs are slow — normal for first run |

---

## What the Backtest Results Mean

**Policy alignment score (0-1):** What fraction of recommended interventions had their associated indicators move in the expected direction. Score >0.5 means better than random. Score >0.7 is strong.

**Weighted alignment:** Same but weights top-ranked recommendations more heavily. If your top recommendation aligned but #8 didn't, weighted alignment is higher than mean alignment. This is what matters.

**Direction accuracy (0-1):** For forecasts, what fraction of predicted deltas had the correct sign. >0.5 = better than coin flip. >0.65 = meaningfully predictive.

**Jaccard stability (0-1):** Overlap between recommendation sets at different base years. >0.8 = very stable. <0.5 = recommendations are sensitive to small data changes (problematic).

---

## What the Ensemble Does

The stacking ensemble trains 4 base models (elastic net, random forest, GBR, HistGBR), collects their out-of-fold predictions via 3-fold CV, then fits a Ridge regression on top to learn the optimal combination weights.

It wins when: base models have complementary errors (one overpredicts where another underpredicts). It loses when: one model dominates and the stacking overhead adds noise.

In the results table, it appears as `ensemble_stack` alongside the other 5 models. The `★` marker indicates whichever model was selected as best for each target.
