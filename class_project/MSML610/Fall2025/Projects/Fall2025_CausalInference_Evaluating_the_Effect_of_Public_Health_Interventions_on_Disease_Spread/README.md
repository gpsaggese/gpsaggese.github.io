
# COVID-19 Vaccination Causal Analysis

A comprehensive causal inference analysis examining the relationship between COVID-19 vaccination rates and pandemic outcomes using global data from Our World in Data.

## Project Overview

This project implements advanced causal inference methods to estimate the effect of COVID-19 vaccination on case rates and mortality, while accounting for complex confounding factors and selection biases in vaccine deployment.

### Key Research Questions
- What is the causal effect of vaccination on COVID-19 case rates?
- How do socioeconomic and demographic confounders influence this relationship?
- What methodological approaches are most robust for causal inference in pandemic data?

## Project Structure

```
Fall2025_CausalInference_Evaluating_the_Effect_of_Public_Health_Interventions_on_Disease_Spread/
├── main.py                         # Main analysis pipeline
├── main.ipynb                      # Complete project (so far) in notebook
├── src/                            # Source code modules
│   ├── data_loader.py              # Data acquisition and loading
│   ├── preprocess.py               # Data cleaning and preprocessing
│   ├── feature_eng.py              # Feature engineering
│   ├── analysis.py                 # Exploratory data analysis
│   ├── causal_analysis.py          # Causal inference methods
│   └── utils.py                    # Utility functions
├── data/
│   ├── processed/                  # Processed datasets
│   └── raw/                        # Raw data (gitignored)
├── results/                        # Output directory
│   ├── output_results.txt          # Complete analysis output
│   └── *.png                       # Visualization plots
└── README.md                       # This file
```

##  Quick Start

### Prerequisites
- Python 3.8+
- Required packages: See `requirements.txt`

### Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python main.py
```

### Required Packages
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- linearmodels
- causalinference
- pathlib
- datetime

### Note : The 'main.ipynb' can be run independently (Optional). Gives entire project updates in single file

##  Data Sources

### Primary Data
- **Our World in Data COVID-19 Dataset**: Comprehensive global COVID-19 statistics
- **Time Period**: Vaccine era (December 2020 onwards)
- **Coverage**: 200+ countries and territories

### Key Variables
- **Outcomes**: Cases per 100k, Deaths per 100k
- **Treatment**: Vaccination percentage
- **Confounders**: Population density, Median age, Hospital beds, GDP per capita
- **Instruments**: Continent-level lagged vaccination rates

##  Methodology

### Causal Inference Framework
1. **Propensity Score Methods**
   - Inverse Probability Weighting (IPW)
   - Propensity Score Matching
   - Stratification

2. **Instrumental Variables**
   - 2-Stage Least Squares (2SLS)
   - Continent-level instruments
   - Lagged vaccination rates

3. **Advanced Diagnostics**
   - Covariate balance assessment
   - Common support evaluation
   - Sensitivity analysis

### Feature Engineering
- **Temporal Features**: 3-week lags and rolling averages
- **Country-Level Aggregation**: Weekly panel data
- **Normalization**: Per-capita metrics for comparability

##  Key Features

### Data Processing Pipeline
- Automated data download from OWID
- Country-level filtering and aggregation
- Missing data handling and imputation
- Temporal alignment and weekly resampling

### Exploratory Analysis
- Correlation matrices and heatmaps
- Country-specific trend visualizations
- Confounder relationship exploration
- Distribution analysis

### Causal Analysis
- Multiple estimation methods comparison
- Robustness checks and diagnostics
- Heterogeneous effects exploration
- Policy implication development

##  Implementation Details

### Main Analysis Components

#### Task 1: Data Acquisition & Preparation
- Downloads and processes raw OWID data
- Creates weekly country-level panel
- Handles missing values and outliers
- Generates key metrics (cases per 100k, vaccination percentage)

#### Task 2: Feature Engineering & Exploration
- Creates lagged variables (3-week lags)
- Computes rolling averages (3-week windows)
- Explores confounder relationships
- Visualizes temporal trends

#### Task 3: Causal Analysis
- Implements instrumental variables regression
- Runs propensity score-based methods
- Performs advanced diagnostics
- Provides causal interpretation

### Output Generation
- **Text Output**: Complete analysis log in `results/output_results.txt`
- **Visualizations**: PNG plots in `results/` directory
- **Processed Data**: Parquet files in `data/processed/`


##  Output Interpretation

### Key Results
- **ATE Estimates**: Average Treatment Effects from multiple methods
- **Balance Diagnostics**: Covariate balance before/after adjustment
- **Sensitivity Analysis**: Robustness to unmeasured confounding
- **Policy Implications**: Interpretable effect sizes and confidence intervals

### Visualization Outputs
- Correlation heatmaps
- Country trend comparisons
- Propensity score distributions
- Causal estimate comparisons

## Methodological Considerations

### Strengths
- Multiple causal inference methods for triangulation
- Comprehensive confounder adjustment
- Global coverage with country-level granularity
- Transparent and reproducible analysis

### Limitations
- Observational data limitations
- Potential for unmeasured confounding
- Country-level aggregation may mask subnational variation
- Vaccine effectiveness may vary by type and variant

---

## Next Steps

### Advanced Causal Methods

#### Instrumental Variables Implementation
- Identify valid instruments: Vaccine supply constraints, delivery timing variations
- Implement 2SLS regression with continent-level instruments
- Test instrument relevance: First-stage F-statistics, partial R-squared
- Validate exclusion restriction: Ensure instruments affect outcomes only through vaccination

#### Difference-in-Differences Analysis
- Identify natural experiments: Country-specific rollout timing variations
- Define treatment and control groups based on vaccination campaign start dates
- Test parallel trends assumption in pre-treatment period
- Estimate dynamic treatment effects over time

#### Sensitivity Analysis Framework
- Rosenbaum bounds for unmeasured confounding
- Placebo tests with pre-vaccination data
- Different model specifications: Varying functional forms and covariate sets
- Subgroup analyses: By continent, income level, healthcare capacity

### Model Validation & Robustness

#### Cross-Validation Procedures
- Leave-one-country-out validation to test generalizability
- Temporal validation: Train on early data, test on later periods
- Bootstrap confidence intervals for all estimates
- Monte Carlo simulations for power analysis

#### Confounding Assessment
- Negative control outcomes: Test effects on unrelated health outcomes
- Positive controls: Validate methods on known relationships
- Mediation analysis: Separate direct and indirect effects
- Time-varying confounding: Address feedback loops between cases and vaccination

### Policy Implications Development

#### Causal Interpretation Framework
- Develop decision rules for interpreting causal estimates
- Quantify confounding strength needed to explain results
- Compare effect sizes to established benchmarks
- Assess practical significance beyond statistical significance

#### Stakeholder Engagement Preparation
- Create policy brief templates for different audiences
- Develop visualization dashboards for interactive exploration
- Prepare sensitivity scenarios for decision-making under uncertainty
- Document methodological limitations and assumptions

### Extended Analysis

#### Heterogeneous Treatment Effects
- Causal forests for discovering effect modifiers
- Interaction analysis by demographic and geographic factors
- Time-varying treatment effects across pandemic waves
- Dose-response relationships with continuous vaccination measures

#### Comparative Effectiveness
- Cross-country comparisons of vaccination strategies
- Vaccine type analysis where data permits
- Combination policies: Vaccination + NPIs interaction effects
- Long-term outcomes: Hospitalizations and mortality analysis

---

**Project Status**: Midterm PR Complete - Foundational analysis implemented and validated

**Ready to proceed with advanced causal methods, comprehensive validation, and policy-relevant analysis in Final Submission.**
```