# Causal Analysis of Economic Factors on Employment Outcomes

This project applies causal inference methods to investigate how macroeconomic indicators causally influence employment outcomes using FRED (Federal Reserve Economic Data). Unlike traditional correlation-based analyses, this work focuses on identifying and quantifying causal relationships through constraint-based discovery algorithms and structural equation modeling.

## Research Question

**How do macroeconomic factors (inflation, unemployment, GDP growth) causally affect employment outcomes (wage growth, employment rates)?**

Traditional econometric models often rely on correlation, but correlation does not imply causation. This project uses causal-learn to discover causal structures from observational data and estimate causal effects, providing interpretable insights into economic relationships.

## Methodology

### Causal Discovery Approach

We employ constraint-based causal discovery algorithms to identify causal pathways:

- **PC Algorithm**: Tests conditional independence to construct causal graphs
- **GES Algorithm**: Score-based search for optimal causal structures  
- **FCI Algorithm**: Handles latent confounders in observational data

### Causal Effect Estimation

Once causal structure is identified, we estimate effects using:

- **Structural Equation Modeling (SEM)**: Quantifies direct and indirect causal effects
- **Regression Adjustment**: Controls for confounders based on discovered causal structure
- **Temporal Analysis**: Examines how causal relationships evolve over time using rolling windows

### Comparative Analysis

We compare causal estimates with predictive machine learning models:

- **Random Forest**: Captures non-linear relationships and feature importance
- **LSTM Networks**: Models temporal dependencies in time series data

This comparison highlights the difference between predictive accuracy and causal interpretability.

## Dataset

**FRED - Federal Reserve Economic Data**

The dataset contains monthly time series data from the Federal Reserve, including:

- **unemployment_rate**: Monthly unemployment rate (%)
- **inflation_rate**: Year-over-year CPI change (%)
- **wage_growth**: Year-over-year average hourly earnings growth (%)
- **gdp_growth**: Real GDP growth rate (%)
- **federal_funds_rate**: Federal funds rate (%)
- **employment_growth**: Year-over-year employment change (%)
- **real_wage_growth**: Inflation-adjusted wage growth (%)

Data is downloaded via the FRED API using the provided script (`data/download_data.py`).

## Installation and Setup

### Requirements

- Python 3.10 or higher
- Docker (optional, for containerized environment)
- FRED API key (free)

### Quick Setup

```bash
# Navigate to project directory
cd class_project/MSML610/Fall2025/Projects/UmdTask83_Fall2025_causal_learn_Economic_Factors_and_Employment_Outcomes

# Install Python dependencies
pip3 install -r requirements.txt

# Get FRED API key (free)
# 1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
# 2. Create account and get API key

# Set API key and download data
export FRED_API_KEY=your_api_key_here
python3 data/download_data.py

# Verify data
ls data/  # Should show economic_data.csv
```

### Docker Setup (Alternative)

```bash
# Build and run container
docker-compose -f docker/docker-compose.yml up --build

# Access Jupyter Lab
# Open http://localhost:8888 in browser
```

## Project Organization

```
.
├── data/                    # Economic data from FRED
│   ├── download_data.py     # Script to download from FRED API
│   └── economic_data.csv    # Downloaded data (gitignored - run download_data.py)
├── utils/                   # Core functionality
│   ├── utils_data_io.py     # Data loading and preprocessing
│   └── utils_post_processing.py  # Causal inference functions
├── notebooks/               # Interactive analysis
│   ├── API.ipynb            # Main analysis notebook
│   └── example.ipynb        # Additional examples
├── scripts/                 # Automated execution
│   ├── api.py               # Pipeline execution
│   └── example.py           # Example workflow
├── pipelines/               # Pipeline definitions
│   └── causal_pipeline.py   # Main causal inference pipeline
├── docs/                    # Documentation
├── docker/                  # Container configuration
├── models/                  # Saved models (generated)
├── outputs/                 # Results and visualizations
│   ├── causal_graphs/       # DAG visualizations
│   └── temporal_analysis/   # Time-varying effects
└── requirements.txt         # Python dependencies
```

## Usage

### Interactive Analysis (Recommended)

Start with the Jupyter notebook for exploratory analysis:

```bash
# Launch Jupyter
jupyter lab notebooks/

# Open notebooks/API.ipynb
```

### Scripted Execution

Run the automated pipeline:

```bash
# Execute full causal inference pipeline
python3 scripts/api.py

# Run example workflow
python3 scripts/example.py
```

### Pipeline Class

```python
from pipelines.causal_pipeline import CausalInferencePipeline

# Initialize pipeline
pipeline = CausalInferencePipeline(algorithm='PC', alpha=0.05)

# Run full analysis
results = pipeline.run_full_analysis()

# Access results
print(f"Discovered edges: {results['edges']}")
print(f"Causal effects: {results['effects']}")
```

## Workflow Overview

### 1. Data Preprocessing

Load FRED data, time-align indicators, and create derived features (real wage growth, employment rate).

### 2. Causal Discovery

Apply PC, GES, or FCI algorithms to identify causal structure. The discovered Directed Acyclic Graph (DAG) represents causal relationships between variables.

### 3. Causal Effect Estimation

Quantify effects using regression adjustment or SEM:
- Direct effects (e.g., unemployment -> wage growth)
- Confidence intervals and statistical significance

### 4. Temporal Analysis

Examine how causal relationships change over time using 36-month rolling windows. This reveals structural breaks during economic crises.

### 5. Model Comparison

Compare causal estimates with Random Forest to understand prediction vs causation.

## Key Research Insights

### Expected Findings

Based on economic theory:

1. **Unemployment -> Wage Growth**: Negative (Phillips Curve - labor slack reduces wage pressure)
2. **Inflation -> Wage Growth**: Positive (nominal wages rise with prices)
3. **Fed Funds Rate -> Inflation**: Negative (monetary policy controls inflation)

### Temporal Dynamics

Causal relationships may vary over time due to:
- Policy changes (monetary/fiscal policy shifts)
- Economic shocks (2008 crisis, COVID-19)
- Structural changes in labor markets

## Configuration

Key parameters in pipeline:

- **Discovery Algorithm**: PC (default), GES, or FCI
- **Significance Threshold**: alpha = 0.05 for independence tests
- **Rolling Window**: 36 months for temporal analysis

## Common Issues

**Import Errors**: Run `pip3 install -r requirements.txt`

**Data Not Found**: Run `export FRED_API_KEY=your_key && python3 data/download_data.py`

**No FRED API Key**: Get free key from https://fred.stlouisfed.org/docs/api/api_key.html

**Docker Problems**: Rebuild with `docker-compose build --no-cache`

## References

- [causal-learn Library](https://causal-learn.readthedocs.io/) - Python causal discovery toolkit
- [FRED Economic Data](https://fred.stlouisfed.org/) - Federal Reserve Economic Data
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

## License

Educational project for MSML610 - Advanced Machine Learning (Fall 2025), University of Maryland
