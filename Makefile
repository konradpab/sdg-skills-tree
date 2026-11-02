.PHONY: install run run-synthetic run-cached test clean lint help preflight verify docker-build docker-run

PYTHON := python3
VENV   := .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install:  ## Install all dependencies
	pip install -e ".[dev]"
	@echo ""
	@echo "Done. Optional extras:"
	@echo "  pip install -e '.[quarto]'      # Quarto report rendering"
	@echo "  pip install -e '.[extensions]'   # NUTS2 maps, Plotly, causal forests"

install-all:  ## Install everything including extensions
	pip install -e ".[dev,quarto,extensions]"

# ── Pipeline ─────────────────────────────────────────────────────────

run:  ## Run full pipeline with REAL data (downloads from APIs)
	$(PYTHON) run_pipeline.py --real

run-synthetic:  ## Run pipeline with synthetic data (no network needed)
	$(PYTHON) run_pipeline.py --synthetic

run-cached:  ## Run pipeline from cached raw data (no re-download)
	$(PYTHON) run_pipeline.py --from-cache

run-fast:  ## Run from cache, skip model training
	$(PYTHON) run_pipeline.py --from-cache --skip-models

# ── Data only ────────────────────────────────────────────────────────

ingest:  ## Download all datasets (no modeling)
	$(PYTHON) -c "from src.ingest.real_data import run_ingestion_real; run_ingestion_real('data_raw')"

test-source:  ## Test a single data source (e.g. make test-source SRC=wdi)
	$(PYTHON) test_source.py $(SRC)

test-sources-all:  ## Test all data source endpoints
	$(PYTHON) test_source.py --all

# ── Manual data steps ────────────────────────────────────────────────
# V-Dem requires manual download (free registration).
# After downloading, place the CSV in data_raw/vdem/

vdem-check:  ## Check if V-Dem data is present
	@if ls data_raw/vdem/*.csv 1>/dev/null 2>&1; then \
	  echo "V-Dem CSV found:"; ls data_raw/vdem/*.csv; \
	else \
	  echo "V-Dem CSV not found."; \
	  echo "1. Register at https://v-dem.net/data/the-v-dem-dataset/"; \
	  echo "2. Download the Country-Year V-Dem Core CSV"; \
	  echo "3. Place it in data_raw/vdem/"; \
	fi

# ── Quality ──────────────────────────────────────────────────────────

test:  ## Run tests
	pytest tests/ -v

lint:  ## Lint with ruff
	ruff check src/ run_pipeline.py

# ── Cleanup ──────────────────────────────────────────────────────────

clean:  ## Remove generated outputs (keeps raw data)
	rm -rf data_interim/* data_processed/* models_saved/* outputs/recommendations/*
	rm -rf reports/figures/* reports/final_report.html
	rm -f pipeline.log

clean-all: clean  ## Remove everything including raw downloads
	rm -rf data_raw/*
	@echo "All data cleared. Run 'make run' to re-download."

# ── Validation ────────────────────────────────────────────────────────

preflight:  ## Run pre-flight environment check
	$(PYTHON) preflight.py

preflight-env:  ## Pre-flight check (skip network)
	$(PYTHON) preflight.py --env-only

verify:  ## Verify outputs from most recent pipeline run
	$(PYTHON) verify_run.py

verify-strict:  ## Verify outputs (strict — fail on warnings)
	$(PYTHON) verify_run.py --strict

# ── Docker ────────────────────────────────────────────────────────────

docker-build:  ## Build Docker image
	docker build -t sdg-skills-tree .

docker-run:  ## Run pipeline in Docker (real data)
	docker compose up pipeline

docker-test:  ## Run tests in Docker
	docker compose run --rm test

docker-preflight:  ## Run pre-flight in Docker
	docker compose run --rm preflight
