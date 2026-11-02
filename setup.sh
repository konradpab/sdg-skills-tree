#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  setup.sh — one-command environment setup
#  Run this in any terminal: bash setup.sh
# ══════════════════════════════════════════════════════════════
set -e

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SDG Skill-Trees — Environment Setup"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Find the right Python ─────────────────────────────────────
# In Codespaces, the Python 3.12 with pip is at /usr/local/bin/python
# On other systems, python3 or python works
if command -v /usr/local/bin/python &>/dev/null; then
    PY=/usr/local/bin/python
elif command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python not found. Install Python 3.10+ first."
    exit 1
fi

echo "Using Python: $PY"
$PY --version
echo ""

# ── Upgrade pip ───────────────────────────────────────────────
echo "Upgrading pip..."
$PY -m pip install --upgrade pip 2>/dev/null || $PY -m ensurepip --upgrade 2>/dev/null && $PY -m pip install --upgrade pip
echo ""

# ── Install all packages directly ─────────────────────────────
echo "Installing packages (this takes 2-3 minutes)..."
$PY -m pip install \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "pyarrow>=14.0" \
    "openpyxl>=3.1" \
    "scikit-learn>=1.3" \
    "xgboost>=2.0" \
    "lightgbm>=4.0" \
    "shap>=0.43" \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    "scipy>=1.11" \
    "networkx>=3.1" \
    "pyyaml>=6.0" \
    "jinja2>=3.1" \
    "requests>=2.31"

echo ""

# ── Install optional visualization packages ───────────────
echo "Installing optional packages (geopandas, plotly)..."
$PY -m pip install "geopandas>=0.14" "plotly>=5.17" 2>/dev/null || echo "(Some optional packages may have failed — not critical)"

# ── Also install the project itself if pyproject.toml exists ──
if [ -f "pyproject.toml" ]; then
    echo "Installing project in editable mode..."
    $PY -m pip install -e . 2>/dev/null || echo "(editable install skipped — packages already installed directly)"
fi

# ── Create directories ────────────────────────────────────────
echo ""
echo "Creating directories..."
mkdir -p data_raw data_interim data_processed models_saved outputs/recommendations reports/figures

# ── Make python/pip point to the right version ────────────────
# This ensures 'python' and 'pip' work in the terminal
if [ "$PY" = "/usr/local/bin/python" ]; then
    # In Codespaces: create aliases in bashrc
    if ! grep -q "alias python=" ~/.bashrc 2>/dev/null; then
        echo 'alias python=/usr/local/bin/python' >> ~/.bashrc
        echo 'alias pip="/usr/local/bin/python -m pip"' >> ~/.bashrc
        echo "Added python/pip aliases to ~/.bashrc"
    fi
    # Also set for the current session
    export PATH="/usr/local/bin:$PATH"
fi

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
$PY -c "
import pandas, numpy, sklearn, xgboost, lightgbm, shap, pyarrow
import matplotlib, seaborn, scipy, networkx, yaml, jinja2, requests
print('✅ All 15 packages installed successfully')
print(f'   pandas {pandas.__version__}, sklearn {sklearn.__version__}, xgboost {xgboost.__version__}')
" || {
    echo "❌ Some packages failed to install. Try running: $PY -m pip install -e ."
    exit 1
}

# ── Check source files ────────────────────────────────────────
echo ""
MISSING=0
for f in src/ingest/real_data.py src/ingest/synthetic_data.py src/clean/normalize.py \
         src/features/engineer.py src/models/train.py src/policy/planner.py \
         src/viz/plots.py src/report/generate_report.py registry/datasets.json; do
    if [ ! -f "$f" ]; then
        echo "⚠  Missing: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "❌ $MISSING source files are missing from the repository."
    echo "   The packages are installed, but the pipeline code is incomplete."
    echo "   Make sure all files from the thesis_deploy folder are on GitHub."
else
    echo "✅ All source files present"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    python run_pipeline.py --synthetic    # test (3 min)"
echo "    python run_pipeline.py --real         # real data (15 min)"
echo "══════════════════════════════════════════════════════════"
echo ""
