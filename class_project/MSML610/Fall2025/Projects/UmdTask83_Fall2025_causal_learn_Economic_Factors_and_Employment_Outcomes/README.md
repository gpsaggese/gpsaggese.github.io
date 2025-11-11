# Causal Analysis of Economic Factors on Employment Outcomes

This project applies causal inference methods to investigate how macroeconomic indicators causally influence employment outcomes using US Labor Statistics data. Unlike traditional correlation-based analyses, this work focuses on identifying and quantifying causal relationships through constraint-based discovery algorithms and structural equation modeling.

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

**US Labor Statistics (FRED/Kaggle)**

The dataset contains monthly time series data from the Bureau of Labor Statistics, including:

- Economic indicators: inflation rates, unemployment rates, GDP growth
- Employment metrics: employment rates, labor force participation
- Wage data: average hourly earnings, wage growth rates
- Industry classifications and seasonal adjustments

Data files are downloaded to the `Data/` directory using the provided download script (`Data/download_data.py`). The dataset includes supporting metadata for series definitions, industry codes, and temporal information.

## Installation and Setup

### Requirements

- Python 3.10 or higher
- Docker (optional, for containerized environment)
- 4GB+ RAM recommended

### Quick Setup

```bash
# Navigate to project directory
cd class_project/MSML610/Fall2025/Projects/UmdTask83_Fall2025_causal_learn_Economic_Factors_and_Employment_Outcomes

# Install Python dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
python Data/download_data.py

# Verify data files
ls Data/  # Should show CSV files and data description
```

**Note**: The large dataset file (`all.data.combined.csv`, ~1.14 GB) is not included in the repository. Run the download script to fetch it from Kaggle. You'll need a Kaggle account and API credentials configured (see [Kaggle API setup](https://www.kaggle.com/docs/api)).

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
├── Data/                    # US Labor Statistics dataset
│   ├── download_data.py     # Script to download dataset from Kaggle
│   ├── all.data.combined.csv  # (downloaded via script, not in repo)
│   ├── ce.*.csv             # Metadata files (downloaded via script)
│   └── data description.txt  # (downloaded via script)
├── utils/                   # Core functionality
│   ├── utils_data_io.py     # Data loading and preprocessing
│   └── utils_post_processing.py  # Causal inference functions
├── notebooks/               # Interactive analysis
│   ├── API.ipynb            # causal-learn API tutorial
│   └── example.ipynb        # Complete workflow demonstration
├── scripts/                 # Automated execution
│   ├── api.py               # Pipeline execution
│   └── example.py           # Example workflow
├── pipelines/               # Pipeline definitions
│   └── causal_pipeline.py   # Main causal inference pipeline
├── docs/                    # Documentation
│   ├── API.md               # API reference
│   └── example.md           # Workflow documentation
├── docker/                  # Container configuration
├── models/                  # Saved models (generated)
├── outputs/                 # Results and visualizations
│   ├── causal_graphs/        # DAG visualizations
│   └── temporal_analysis/    # Time-varying effects
└── requirements.txt         # Python dependencies
```

## Usage

### Interactive Analysis (Recommended)

Start with the Jupyter notebooks for exploratory analysis:

```bash
# Launch Jupyter
jupyter lab notebooks/

# Or in Docker
docker exec -it causal-econ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Notebooks:**
- `notebooks/API.ipynb`: Learn the causal-learn API with examples
- `notebooks/example.ipynb`: Complete end-to-end causal analysis workflow

### Scripted Execution

Run the automated pipeline:

```bash
# Execute full causal inference pipeline
python scripts/api.py

# Run example workflow
python scripts/example.py
```

## Workflow Overview

### 1. Data Preprocessing

Time-align economic indicators and employment outcomes, handle missing values, and create derived features (e.g., wage growth rates, inflation-adjusted metrics).

### 2. Causal Discovery

Apply PC, GES, or FCI algorithms to identify causal structure. The discovered Directed Acyclic Graph (DAG) represents causal relationships between variables.

### 3. Causal Effect Estimation

Use Structural Equation Modeling to quantify:
- Direct effects (e.g., inflation → wage growth)
- Indirect effects (e.g., inflation → unemployment → wage growth)
- Confidence intervals and statistical significance

### 4. Temporal Analysis

Examine how causal relationships change over time using rolling window analysis. This reveals structural breaks and time-varying effects.

### 5. Model Comparison

Compare causal estimates with predictive ML models to understand the trade-off between interpretability (causal) and accuracy (predictive).

## Key Research Insights

### Expected Findings

Based on economic theory, we expect to find:

1. **Inflation → Wage Growth**: Negative causal effect (inflation erodes purchasing power)
2. **Unemployment → Wage Growth**: Negative causal effect (labor market slack reduces wage pressure)
3. **GDP Growth → Employment Rate**: Positive causal effect (economic expansion creates jobs)

### Temporal Dynamics

Causal relationships may vary over time due to:
- Policy changes (monetary/fiscal policy shifts)
- Economic shocks (recessions, crises)
- Structural changes in labor markets

Rolling window analysis helps detect these temporal variations.

## Technical Details

### Causal Discovery Algorithms

**PC Algorithm** (Peter-Clark):
- Constraint-based method using conditional independence tests
- Suitable for continuous variables (Fisher's Z-test)
- Default algorithm for this project

**GES Algorithm** (Greedy Equivalence Search):
- Score-based approach optimizing BIC or BDeu scores
- Alternative when constraint-based methods are insufficient

**FCI Algorithm** (Fast Causal Inference):
- Extends PC to handle latent confounders
- Produces Partial Ancestral Graphs (PAGs) with bidirected edges

### Effect Estimation Methods

**Structural Equation Modeling (SEM)**:
- Estimates path coefficients in the causal graph
- Provides confidence intervals and significance tests
- Handles both direct and indirect effects

**Regression Adjustment**:
- Controls for confounders identified in causal discovery
- Simpler alternative to SEM for linear relationships

## Development Status

**Current Phase**: Foundation established

**Completed**:
- Project structure and organization
- Utility modules for data I/O and causal inference
- Documentation framework
- Docker environment setup

**In Progress**:
- Data preprocessing implementation
- Causal discovery algorithm integration
- Effect estimation methods

**Planned**:
- Temporal analysis implementation
- ML model comparisons
- Visualization and reporting

## Configuration

Key parameters are set in pipeline scripts:

- **Discovery Algorithm**: PC (default), GES, or FCI
- **Significance Threshold**: α = 0.05 for independence tests
- **Rolling Window**: 24 months for temporal analysis
- **Model Hyperparameters**: Defined in respective training scripts

## Common Issues

**Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

**Data Not Found**: Run `python Data/download_data.py` to download the dataset from Kaggle. Ensure you have Kaggle API credentials configured.

**Docker Problems**: Rebuild container with `docker-compose build --no-cache`

**Causal Discovery Fails**: Check data quality, missing values, and sample size (minimum ~100 observations recommended)

## Documentation

- **API Reference**: `docs/API.md` - Detailed causal-learn API documentation
- **Workflow Guide**: `docs/example.md` - Step-by-step analysis walkthrough
- **Interactive Tutorials**: `notebooks/` - Hands-on Jupyter notebooks

## Academic Context

This project demonstrates the application of modern causal inference methods to economic data. Unlike traditional econometric approaches that assume causal structure a priori, causal discovery algorithms learn structure from data, making this approach particularly valuable when theoretical knowledge is incomplete.

The combination of causal discovery and effect estimation provides:
- **Interpretability**: Clear causal pathways and effect sizes
- **Robustness**: Methods account for confounders
- **Temporal Insights**: Understanding of how relationships evolve

## References

- [causal-learn Library](https://causal-learn.readthedocs.io/) - Python causal discovery toolkit
- [FRED Economic Data](https://fred.stlouisfed.org/) - Federal Reserve Economic Data
- [US Labor Statistics](https://www.kaggle.com/datasets/bls/employment) - Bureau of Labor Statistics data
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

## License

Educational project for MSML610 - Advanced Machine Learning (Fall 2025), University of Maryland

---

**Project Status**: Active Development  
**Last Updated**: Phase 1 Complete
