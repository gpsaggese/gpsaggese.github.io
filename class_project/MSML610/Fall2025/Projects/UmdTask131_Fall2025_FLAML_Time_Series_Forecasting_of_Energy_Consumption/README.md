# FLAML AutoML Project: API Tutorial & Energy Forecasting Application

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FLAML](https://img.shields.io/badge/FLAML-2.1+-green.svg)](https://microsoft.github.io/FLAML/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides a comprehensive exploration of **FLAML (Fast and Lightweight AutoML)** through two complementary Jupyter notebooks:

| Notebook | Purpose | Focus |
|----------|---------|-------|
| **FLAML_API.ipynb** | Tool Introduction | Learn FLAML's interface, capabilities, and usage patterns |
| **FLAML_Example.ipynb** | Real-World Application | Apply FLAML to energy consumption forecasting |

Together, these notebooks offer both **conceptual understanding** and **practical implementation** of AutoML for time series forecasting.

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd UMDTASK131_FALL2025_FLAML

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the UCI Household Electric Power Consumption dataset and place it in the `data/` folder:

```
data/household_power_consumption.txt
```

**Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

### 3. Run Notebooks

```bash
# Option 1: Jupyter Notebook
jupyter notebook

# Option 2: JupyterLab
jupyter lab

# Option 3: Docker
cd docker/ && docker compose up --build
```

### 4. Explore Results

After running `FLAML_Example.ipynb`:
- Visualizations saved to `outputs/` directory
- Interactive dashboard: `streamlit run dashboard.py`

---

## Project Structure

```
UMDTASK131_FALL2025_FLAML/
│
├── 📓 FLAML_API.ipynb           # Tutorial notebook - Learn FLAML
├── 📓 FLAML_Example.ipynb       # Application notebook - Energy forecasting
│
├── 📄 FLAML_API.md              # Detailed README for API notebook
├── 📄 FLAML_Example.md          # Detailed README for Example notebook
├── 📄 README.md                 # This file - Project overview
│
├── 🐍 utils.py                  # Reusable utility functions
├── 🐍 dashboard.py              # Streamlit interactive dashboard
├── 📋 requirements.txt          # Python dependencies
│
├── 📁 data/                     # Dataset directory
│   └── household_power_consumption.txt
│
├── 📁 outputs/                  # Generated outputs
│   ├── *.png                    # Visualizations (6 files)
│   ├── *.csv                    # Data files (3 files)
│   └── summary.json             # Results summary
│
└── 📁 docker/                   # Docker configuration
    ├── Dockerfile
    ├── docker-compose.yml
    └── run_jupyter_server.sh
```

---

## Notebooks Overview

### 📘 FLAML_API.ipynb — Learn the Tool

**Purpose:** Understand FLAML's interface and capabilities without project-specific complexity.

**What You'll Learn:**
- AutoML concepts and FLAML's design philosophy
- Classification, regression, and forecasting examples
- Model persistence and pipeline integration
- Best practices and limitations

**Key Topics:**

| Section | Content |
|---------|---------|
| Introduction | AutoML concepts, FLAML overview |
| Example 1 | Classification (Iris dataset) |
| Example 2 | Regression (California Housing) |
| Example 3 | Time Series Forecasting (Synthetic) |
| Integration | Scikit-learn pipeline integration |
| Best Practices | Tips and recommendations |

**Runtime:** ~5 minutes

📖 **Detailed Documentation:** [FLAML_API.md](FLAML_API.md)

---

### 📗 FLAML_Example.ipynb — Apply the Tool

**Purpose:** Apply FLAML to a real-world energy consumption forecasting problem.

**What You'll Achieve:**
- 96.64% prediction accuracy with XGBoost
- Comparison of 7 models (4 FLAML + Prophet + ARIMA + Ensemble)
- 41 engineered features from raw data
- Production-ready forecasting pipeline

**Key Results:**

| Model | Test RMSE | Test MAPE | Accuracy |
|-------|-----------|-----------|----------|
| **FLAML (XGBoost)** | **0.0388** | **3.36%** | **96.64%** |
| FLAML (LightGBM) | 0.0422 | 3.59% | 96.41% |
| Prophet | 0.2585 | 22.14% | 77.86% |
| ARIMA | 0.3344 | 37.86% | 62.14% |

**Sections:**

| # | Section | Content |
|---|---------|---------|
| 1 | Setup | Configuration, imports |
| 2 | EDA | Data exploration, visualizations |
| 3 | Preprocessing | Resampling, cleaning |
| 4 | Feature Engineering | 41 features created |
| 5 | Train-Test Split | Temporal split (80/20) |
| 6 | Model Training | FLAML, Prophet, ARIMA |
| 7 | Advanced Analysis | Feature importance, ensemble, rolling eval |
| 8 | Summary | Business impact, conclusions |

**Runtime:** ~10-15 minutes

📖 **Detailed Documentation:** [FLAML_Example.md](FLAML_Example.md)

---

## Key Features

### 🤖 AutoML with FLAML

- Automated model selection from LightGBM, XGBoost, Random Forest, Extra Trees
- Hyperparameter tuning with cost-aware optimization
- Time budget control (120 seconds per model)
- 3-fold cross-validation

### 📊 Comprehensive Feature Engineering

| Category | Count | Examples |
|----------|-------|----------|
| Temporal | 13 | day_of_week, month, season, cyclical encodings |
| Lag | 6 | lag_1, lag_7, lag_30 |
| Rolling | 12 | rolling_mean_7, rolling_std_14 |
| EMA | 2 | ema_7, ema_30 |
| Difference | 2 | diff_1, diff_7 |
| **Total** | **41** | |

### 📈 Multiple Baselines

- **Prophet:** Facebook's decomposition-based forecasting
- **ARIMA:** Classical statistical baseline
- **Ensemble:** 60% FLAML + 40% Prophet weighted combination

### 🎯 Bonus Features

- ✅ Rolling forecast evaluation (30-day windows)
- ✅ Ensemble forecasting (60-40 weighted)
- ✅ Feature importance analysis
- ✅ Seasonality & volatility analysis

### 🖥️ Interactive Dashboard

```bash
streamlit run dashboard.py
```

Explore results interactively with:
- Model comparison charts
- Prediction visualizations
- Feature importance explorer
- Performance metrics

---

## Requirements

### Python Version

- Python 3.8 or higher

### Core Dependencies

```
flaml>=2.1.0
prophet>=1.1.5
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
```

### Installation

```bash
pip install -r requirements.txt
```

---

## Output Files

After running `FLAML_Example.ipynb`, the following files are generated:

### Visualizations (6 PNG files)

| File | Description |
|------|-------------|
| `exploratory_analysis.png` | 9-panel EDA visualization |
| `cleaned_timeseries.png` | Daily consumption with moving averages |
| `feature_correlations.png` | Feature correlation heatmap |
| `train_test_split.png` | Train/test split visualization |
| `feature_importance.png` | XGBoost feature importance |
| `rolling_forecast.png` | Rolling evaluation results |

### Data Files (3 CSV files)

| File | Description |
|------|-------------|
| `predictions.csv` | Actual vs predicted for all models |
| `model_comparison.csv` | Final model performance metrics |
| `flaml_candidates_comparison.csv` | 4 FLAML models comparison |

### Summary (1 JSON file)

| File | Description |
|------|-------------|
| `summary.json` | Complete project results and configuration |

---

## Utility Functions

The `utils.py` module provides reusable functions:

```python
from utils import (
    add_temporal_features,    # 13 calendar features
    add_lag_features,         # 6 lag features
    add_rolling_features,     # 12 rolling statistics
    add_ema_features,         # 2 EMA features
    calculate_metrics         # RMSE, MAE, MAPE, R²
)
```

---

## Docker Support

Run the entire project in a containerized environment:

```bash
cd docker/
docker compose up --build
```

Access JupyterLab at: `http://localhost:8888`

---

## Educational Value

### Learning Path

```
1. Start with FLAML_API.ipynb
   └── Understand FLAML interface and concepts
   
2. Progress to FLAML_Example.ipynb
   └── Apply knowledge to real-world forecasting
   
3. Explore dashboard.py
   └── Interactive result exploration
   
4. Extend with your own data
   └── Use utils.py for new projects
```

### Concepts Covered

| Topic | FLAML_API | FLAML_Example |
|-------|-----------|---------------|
| AutoML basics | ✅ | ✅ |
| Classification | ✅ | |
| Regression | ✅ | |
| Time Series Forecasting | ✅ | ✅ |
| Feature Engineering | | ✅ |
| Model Comparison | | ✅ |
| Rolling Evaluation | | ✅ |
| Business Impact | | ✅ |

### Skills Developed

- Automated machine learning workflows
- Time series preprocessing and feature engineering
- Model evaluation and comparison
- Python libraries: pandas, scikit-learn, FLAML, Prophet
- Documentation and reproducibility best practices

---

## Project Requirements Status

### Core Requirements

| Requirement | Status |
|-------------|--------|
| Data Preparation (cleaning, resampling, missing values) | ✅ |
| Feature Engineering (lags, rolling, temporal, EMA) | ✅ |
| Model Training with FLAML (LightGBM, XGBoost, RF, Extra Trees) | ✅ |
| Model Comparison (RMSE, MAPE, R² for all models) | ✅ |
| Visualization (predicted vs actual) | ✅ |
| Analysis (seasonality and volatility) | ✅ |

### Bonus Requirements ✅

| Requirement | Status |
|-------------|--------|
| Rolling forecast evaluation | ✅ |
| Ensemble forecasting | ✅ |

---

## References

### Academic Papers

1. Wang, C., et al. (2021). "FLAML: A Fast and Lightweight AutoML Library." *MLSys 2021*.
2. Taylor, S. J., & Letham, B. (2018). "Forecasting at scale." *The American Statistician*.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *KDD 2016*.
4. Ke, G., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." *NIPS 2017*.

### Documentation

- [FLAML Documentation](https://microsoft.github.io/FLAML/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Dataset

- [UCI ML Repository: Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

---

## License

This project is licensed under the MIT License.

---

## Author

**Anisha Katiyar**  
MSML610 - Advanced Machine Learning  
University of Maryland  
Fall 2025

---

## Acknowledgments

- **Microsoft Research** — FLAML library development
- **Meta Research** — Prophet library development
- **UCI ML Repository** — Dataset hosting
- **Course Instructors & TAs** — MSML610 guidance

---