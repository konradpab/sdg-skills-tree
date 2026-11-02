# Running the Pipeline in GitHub Codespaces

This guide assumes you have already uploaded the code to a GitHub repository
and that you have zero experience with terminals or command lines. Every
click and every command is described exactly.

Total time: about 20–30 minutes (most of it is waiting).

---

## Step 1: Open Your Repository on GitHub

1. Open your browser (Chrome, Firefox, Safari — any is fine)
2. Go to `https://github.com/<your-username>/sdg-skills-tree`
   (replace `<your-username>` with your actual GitHub username)
3. You should see a page listing your project files:
   `config.yaml`, `run_pipeline.py`, `src/`, `README.md`, etc.

If you don't see these files, the upload didn't work. Go back and check.

---

## Step 2: Create a Codespace

A Codespace is a computer in the cloud that GitHub gives you for free.
It runs in your browser — you don't install anything on your machine.

1. On your repository page, find the green button that says **`<> Code`**.
   It's near the top right, above the file listing.

2. Click that green **`<> Code`** button. A dropdown appears.

3. In the dropdown, you'll see two tabs at the top: **"Local"** and
   **"Codespaces"**. Click the **"Codespaces"** tab.

4. Click the green button **"Create codespace on main"**.

5. **Wait.** A new browser tab opens. You'll see a loading screen that
   says "Setting up your codespace". This takes **2–4 minutes**. Don't
   close the tab. It's installing Python and all the project's packages
   automatically.

6. When it's done, you'll see something that looks like VS Code (a code
   editor) inside your browser. There are three main areas:
   - **Left sidebar:** a file explorer showing your project files
   - **Main area:** a welcome tab or an empty editor
   - **Bottom panel:** a terminal (a black area where you type commands)

If you don't see the terminal at the bottom, press **Ctrl+`** (that's
the backtick key, to the left of the 1 key) or go to the top menu:
**Terminal → New Terminal**.

---

## Step 3: Verify the Setup Worked

In the terminal at the bottom (the black area with a blinking cursor),
type this command and press **Enter**:

```
python preflight.py --env-only
```

**What you should see:** A checklist with lots of green check marks (✓):

```
  ✓ Python 3.12.x
  ✓ pandas (2.x.x)
  ✓ numpy (...)
  ✓ pyarrow (...)
  ✓ scikit-learn (...)
  ✓ xgboost (...)
  ✓ lightgbm (...)
  ✓ shap (...)
  ... (all green)
  ✓ Project structure
  ✓ Configuration
```

**If you see red ✗ marks on packages:** The automatic install didn't
finish. Run this command and wait for it to complete:

```
pip install -e .
```

This takes 2–3 minutes. Then run `python preflight.py --env-only` again.

---

## Step 4: Test That the Pipeline Works (Synthetic Data)

Before downloading real data, let's make sure everything runs. Type:

```
python run_pipeline.py --synthetic
```

Press **Enter** and **wait about 3 minutes**. You'll see a lot of text
scrolling by. This is normal. The important thing is that it ends with
something like:

```
======================================================================
  PIPELINE COMPLETE
======================================================================
  Time: 180.5s
```

**If you see errors instead:** copy the last 20 lines of the error
and bring them back to this conversation. We'll fix it.

**If it completed successfully:** great! Let's verify the outputs:

```
python verify_run.py
```

You should see mostly green ✓ marks. The pipeline works.

---

## Step 5: Check That Data Sources Are Reachable

Now let's make sure the real-world data APIs are accessible:

```
python test_source.py --all
```

This tests each of the 9 data sources. It takes about a minute. You want
to see `✓ OK` for most of them:

```
  ✓ wdi: OK
  ✓ wgi: OK
  ✓ unsd_sdg: OK
  ✓ ember: OK
  ✓ primap: OK
  ✓ swiid: OK
  ✓ carbon_pricing: OK
  ✓ ucdp: OK
  ⚠ vdem: EMPTY (that's OK — it needs a manual download)
```

**If any source besides V-Dem shows ✗ FAILED:**

That source is temporarily down. Don't worry — we can disable it. You
would open `config.yaml` and set that source to `false`. But let's try
running first and see if it works.

---

## Step 6: (Optional) Add V-Dem Data

V-Dem is the only dataset that can't be auto-downloaded because it
requires free registration. **You can skip this step** — the pipeline
works fine without it. You just get slightly fewer governance indicators.

**If you want to include it:**

1. On your regular computer (not in the Codespace), go to:
   https://v-dem.net/data/the-v-dem-dataset/
2. Click "Download" → register for a free account
3. Download the **"Country-Year: V-Dem Core"** CSV file (~200 MB)
4. Go back to your Codespace in the browser
5. In the terminal, type:
   ```
   mkdir -p data_raw/vdem
   ```
6. In the left sidebar (file explorer), find the `data_raw/vdem/` folder
7. **Drag and drop** the downloaded V-Dem CSV file from your computer
   into that `data_raw/vdem/` folder in the sidebar

**If you're skipping V-Dem**, open `config.yaml` in the Codespace
(click it in the left sidebar) and change line 25 from `vdem: true`
to `vdem: false`. Press **Ctrl+S** to save.

---

## Step 7: Run the Pipeline on Real Data

This is the main event. Type:

```
python run_pipeline.py --real
```

Press **Enter** and wait. **This takes 5–15 minutes.** Here's what
happens and roughly when:

**Minutes 0–8: Downloading data**
```
[UNSD] Fetching elec_access (EG_ELC_ACCS)...
  elec_access: 1847 rows
[EMBER] Downloading from https://ember-energy.org/...
[OK] ember: 18432 rows
[WGI] Fetching governance indicators...
  Fetching GE.EST → gov_effectiveness...
```

You'll see each dataset being fetched one by one. Some are fast (World
Bank takes seconds), some are slow (Ember and PRIMAP can take minutes).

**If you see a `[FAIL]` message** for one source — that's OK. The
pipeline continues with whatever it got. It only stops if ALL sources fail.

**Minutes 8–10: Processing**
```
[CLEAN] Normalized panel: 24350 rows, 187 countries
[FEATURES] 158 features generated
```

**Minutes 10–14: Training models**
```
Training for: Δ Renewables Share (1y)
  Best model: hist_gb (val_mae=0.42)
Training for: Δ CO₂ per capita (1y)
  Best model: gradient_boosting (val_mae=0.31)
```

**Minutes 14–15: Generating report**
```
[VIZ] Generating figures...
[REPORT] Saved to reports/final_report.html

PIPELINE COMPLETE
  Time: 612.4s
```

---

## Step 8: Check the Results

Run:

```
python verify_run.py
```

**What you want to see:**

```
  ✓ data_processed/master_panel.csv (5,231,000 bytes)
  ✓ models_saved/training_results.json
  ✓ reports/final_report.html (1,800,000 bytes)
  ✓ POL roadmap: 10 interventions
  ✓ DEU roadmap: 11 interventions
  ✓ IND roadmap: 10 interventions
  ✓ Trained 10 targets
  ✓ Models beat naive baseline on 8/10 targets
  ✓ Panel covers 150 countries
```

The exact numbers will differ, but you want mostly green ✓ marks and the
panel covering 100+ countries (that confirms real data was ingested, not
the 50-country synthetic data).

---

## Step 9: View the Report

The report is an HTML file. Here's how to see it:

### Method A: Download It (Simplest)

1. In the left sidebar, navigate to `reports/final_report.html`
2. **Right-click** on the file
3. Click **"Download..."**
4. Open the downloaded file in your browser (double-click it)

### Method B: Preview Inside the Codespace

1. In the left sidebar, click on `reports/final_report.html`
2. The file opens in the editor (you'll see HTML code — that's normal)
3. Press **Ctrl+Shift+P** (or Cmd+Shift+P on Mac) to open the
   Command Palette (a search bar appears at the top)
4. Type **"Simple Browser"** and select **"Simple Browser: Show"**
5. In the URL bar that appears, type:
   ```
   /workspaces/sdg-skills-tree/reports/final_report.html
   ```
   (This might not work in all Codespace configurations — Method A is
   more reliable.)

### What the Report Contains

The report has 7 sections:
1. **Motivation** — why this work matters
2. **Data & Provenance** — table of all datasets, coverage heatmap
3. **Methods** — how features, models, and the planner work
4. **Results** — model performance tables and charts
5. **Case Studies** — Poland, Germany, India with roadmaps
6. **Discussion** — limitations, fairness, policy implications
7. **Conclusion** — summary and next steps

---

## Step 10: Download All Results

You probably want to save everything to your computer.

### Download the Report Only

Right-click `reports/final_report.html` in the sidebar → **Download**.

### Download Everything as a Zip

In the terminal, type these commands one at a time:

```
tar czf results.tar.gz reports/final_report.html reports/figures/ outputs/recommendations/ data_processed/master_panel.csv models_saved/training_results.json models_saved/feature_importances.json data_interim/coverage_report.json pipeline.log
```

Then in the left sidebar, find `results.tar.gz` at the top of the file
list, right-click it → **Download**.

You can unzip it on your computer with:
- **macOS/Linux:** double-click the file, or `tar xzf results.tar.gz`
- **Windows:** use 7-Zip or WinRAR

---

## Step 11: Make Changes and Re-Run

### Change Case Study Countries

1. In the left sidebar, click `config.yaml` to open it
2. Find this line (around line 17):
   ```yaml
   case_study_countries: ["POL", "DEU", "IND"]
   ```
3. Change the country codes. Some examples:
   - `["POL", "DEU", "BRA"]` — Poland, Germany, Brazil
   - `["NGA", "ZAF", "KEN"]` — Nigeria, South Africa, Kenya
   - `["POL", "NLD", "CHL"]` — Poland, Netherlands, Chile
4. Press **Ctrl+S** to save
5. In the terminal, run:
   ```
   python run_pipeline.py --from-cache --skip-models
   ```
   This takes only about 30 seconds because it reuses the downloaded
   data and trained models — it only re-generates recommendations and
   the report.

### Adjust How Interventions Are Ranked

1. Open `config.yaml`
2. Find the `planner: weights:` section (around line 126):
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
3. Change the numbers. They should add up to 1.0. For example, to
   prioritize CO₂ reduction more heavily:
   ```yaml
       w_sdg:         0.20
       w_co2:         0.40
       w_equity:      0.10
       w_feasibility: 0.10
       w_speed:       0.10
       w_cost:        0.10
   ```
4. Press **Ctrl+S** to save
5. In the terminal, run:
   ```
   python run_pipeline.py --from-cache --skip-models
   ```

### Save Your Changes Back to GitHub

After making changes you want to keep:

1. In the left sidebar, look for the **Source Control** icon (it looks
   like a branch/fork symbol, third icon from top). It might show a
   number badge indicating changed files.
2. Click it. You'll see a list of changed files.
3. In the text box that says **"Message"** at the top, type a short
   description like: `Changed case study countries to POL, NLD, BRA`
4. Click the **"Commit"** button (or the ✓ checkmark)
5. A prompt may ask "Stage all changes?" — click **"Yes"**
6. Click **"Sync Changes"** (or the cloud icon with an arrow)

Your changes are now saved on GitHub.

---

## Step 12: Stop the Codespace (Save Money)

GitHub gives you 120 free core-hours per month. A running Codespace uses
them even when you're not typing. **Always stop it when you're done.**

### Method A: From the Codespace

1. Click the **"Codespaces"** text in the very bottom-left corner of
   the screen (in the green/blue bar)
2. Select **"Stop Current Codespace"**

### Method B: From GitHub

1. Go to https://github.com/codespaces
2. Find your codespace in the list
3. Click the **"..."** menu next to it
4. Click **"Stop codespace"**

### Restarting Later

When you want to continue:
1. Go to https://github.com/codespaces
2. Click on your existing codespace
3. It restarts in about 30 seconds
4. Your files and terminal history are still there

**Important:** Codespaces are automatically deleted after 30 days of
inactivity. Download your results before that (Step 10), or push your
changes to GitHub (Step 11) to keep everything safe.

---

## Troubleshooting

### "The terminal shows `$` but I don't know where to type"

Click anywhere inside the black terminal panel at the bottom. You should
see a blinking cursor after the `$` sign. Type your command there and
press Enter.

### "I don't see a terminal at the bottom"

Press **Ctrl+`** (backtick, the key to the left of 1). Or go to the
top menu bar: **Terminal → New Terminal**.

### "pip install -e . gives errors about gcc or compilation"

Run this first, then retry:
```
sudo apt-get update && sudo apt-get install -y gcc g++ cmake
pip install -e .
```

### "A data source failed during the real run"

Check which sources succeeded:
```
python -c "import json; d=json.load(open('data_raw/ingestion_manifest.json')); print('Got:', d.get('sources_ingested',[])); print('Missed:', d.get('sources_skipped',[]))"
```

If only 1-2 sources failed, the pipeline still works with the others.
If you want to retry just the failed source:
```
python test_source.py <source_name>
```
(replace `<source_name>` with the one that failed, e.g. `ember` or `primap`)

If it still fails, disable it in `config.yaml` by setting it to `false`,
then re-run:
```
python run_pipeline.py --real
```

### "The pipeline ran but verify_run.py shows only 50 countries"

This means it used synthetic data instead of real data. Make sure you
ran with `--real`, not `--synthetic`:
```
python run_pipeline.py --real
```

### "I accidentally closed the browser tab"

Go to https://github.com/codespaces — your codespace is still running.
Click on it to reconnect. Nothing is lost.

### "The Codespace is taking forever to build"

The first time takes 3–5 minutes because it's installing all Python
packages. If it takes more than 10 minutes, something is wrong. Delete
it from https://github.com/codespaces and create a new one.

### "I want to start completely over"

1. Go to https://github.com/codespaces
2. Click **"..."** next to your codespace → **"Delete"**
3. Go back to your repo and create a new Codespace (Step 2)

---

## Quick Reference Card

Save this for later — these are the only commands you need:

```bash
# Test everything works (no real data needed)
python run_pipeline.py --synthetic

# Run on real data (downloads everything, ~15 min)
python run_pipeline.py --real

# Re-run after changing config (fast, ~30 sec)
python run_pipeline.py --from-cache --skip-models

# Check if results are good
python verify_run.py

# Test if data sources are online
python test_source.py --all

# Package results for download
tar czf results.tar.gz reports/ outputs/ data_processed/master_panel.csv
```
