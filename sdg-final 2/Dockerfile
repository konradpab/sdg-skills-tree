FROM python:3.12-slim

LABEL maintainer="konrad" \
      description="SDG Skill-Trees: ML Pipeline for Decarbonization Roadmaps"

# System dependencies for openpyxl, matplotlib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /thesis

# Install Python deps first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir \
      pandas>=2.0 numpy>=1.24 pyarrow>=14.0 openpyxl>=3.1 \
      scikit-learn>=1.3 xgboost>=2.0 lightgbm>=4.0 shap>=0.43 \
      matplotlib>=3.7 seaborn>=0.12 scipy>=1.11 networkx>=3.1 \
      pyyaml>=6.0 jinja2>=3.1 requests>=2.31

# Copy project
COPY . .

# Create data directories
RUN mkdir -p data_raw data_interim data_processed models_saved \
    outputs/recommendations reports/figures

# Default: run full pipeline on real data
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--real"]
