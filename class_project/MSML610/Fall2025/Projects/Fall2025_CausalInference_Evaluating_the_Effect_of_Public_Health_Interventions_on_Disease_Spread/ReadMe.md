# CausaL Analysis - Evaluating the Effect of Public Health Interventions on Disease Spread  

A Complete Causal Inference Analysis Using Global COVID-19 Data  

This project implements a fully automated, end-to-end causal inference pipeline to study the relationship between COVID-19 vaccination rates and disease spread using real-world data from **Our World in Data (OWID)**.  
The pipeline includes data acquisition, preprocessing, weekly panel construction, feature engineering, causal modeling, robustness checks, validation procedures, and healthcare system comparisons all executed through a single unified script.

---

## 1. Project Overview

This study investigates how vaccination levels influence COVID-19 outcomes while adjusting for demographic, economic, and healthcare-related confounding factors. Multiple causal inference methods are used to ensure triangulation and methodological robustness.

### Core objectives:
- Estimate the causal effect of vaccination on COVID-19 case rates.
- Identify and control for major confounders.
- Assess sensitivity, heterogeneity, and robustness.
- Validate model performance across countries and time periods.
- Examine how healthcare system types moderate effects.

The entire workflow is reproducible and produces structured outputs, figures, and a complete analysis log.

---

## 2. Project Structure

```
Fall2025_CausalInference_Evaluating_the_Effect_of_Public_Health_Interventions_on_Disease_Spread/
│
├── Causal_Inference.example.py                          # Unified pipeline runner
├── Causal_Inference.example.ipynb                       # Complete project notebook (created for easy explanation)
├── Causal_Inference.API.ipynb                           # API usage notebook
├── Causal_Inference.example.md                          # Example application using API layer
├── Causal_Inference.API.md                              # API documentation
├── docker_runner.sh                                     # Automates building, running, entering, and cleaning containers
├── Dockerfile                                           # Defines the container environment used to run full analysis
├── license_dataset.md                                   # Dataset usage terms and attribution
├── requirements.txt                                     # All Python dependencies
├── README.md                                            # This file
│
├── src/                                                 # Core modular codebase
│   ├── data_loader.py                                   # Data acquisition (OWID)
│   ├── preprocess.py                                    # Cleaning, weekly panel creation
│   ├── feature_eng.py                                   # Lag/rolling features
│   ├── causal_analysis.py                               # Full analysis workflow
│   ├── utils.py                                         # Logging + Tee + helpers
│   
├── data/
│   ├── raw/                                             # Auto-downloaded OWID dataset
│   └── processed/                                       # Cleaned output datasets
│
└── results/
    ├── output_results.txt                               # Full terminal log (100% captured)
    └── plots/                                           # All generated figures
```

---

## 3. Pipeline Execution

The entire analysis runs through a single command:

### Prerequisites
- Python 3.8+
- Required packages: See `requirements.txt`

### Access the project folder

Dowload or clone the repo to get access to the project folder from https://github.com/gpsaggese-org/umd_classes/tree/UmdTask73_Fall2025_CausalInference_Evaluating_the_Effect_of_Public_Health_Interventions_on_Disease_Spread/class_project/MSML610/Fall2025/Projects/Fall2025_CausalInference_Evaluating_the_Effect_of_Public_Health_Interventions_on_Disease_Spread

### (Option 1) Quick Start - Docker

1. Move to correct working directory
```bash
cd #Enter the path to the project folder here#
```

2. Install dependencies:
```bash
./docker_runner.sh run   
```
The above code will automatically run the entire project

### (Option 2) Quick Start - Direct .py execution

1. Move to correct working directory
```bash
cd #Enter the path to the project folder here#
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python Causal_Inference.example.py
```

The script automatically:

1. Downloads raw OWID data  
2. Cleans and validates the dataset  
3. Builds weekly country-level panels  
4. Engineers lagged and rolling features  
5. Runs the full causal inference pipeline  
6. Performs sensitivity + robustness checks  
7. Conducts model validation  
8. Generates all plots  
9. Saves a complete log of all printed results  

---

## 4. What has been Implemented:

Task 1: Data Acquisition and Preparation
- Automated download of the OWID dataset
- Extensive cleaning and filtering
- Construction of a validated weekly panel dataset
- Standardized per-capita normalization and treatment variable definition

Task 2: Feature Engineering and Exploration
- Temporal lag construction
- Rolling averages for cases, deaths, and vaccination
- Confounder exploration and correlation assessment
- Continental and country-level vaccination trend analysis
- Multiple diagnostic visualizations

Task 3: Causal Inference Methods
- Propensity score estimation and matching
- Inverse probability weighting
- Covariate balance diagnostics
- Instrumental variable analysis using continental vaccination intensity
- Sensitivity and robustness checks
- Interim causal interpretation

Task 4: Advanced Methods, Validation, and Heterogeneous Effects
- Difference-in-Differences estimation
- Model validation procedures
- Subgroup analyses across continents and healthcare capacity
- Consolidated causal interpretation and policy-relevant conclusions

Bonus Task: Healthcare System Capacity and Policy Analysis
- Extended analysis linking vaccination effects to healthcare infrastructure differences
- Comparison of causal estimates across countries grouped by hospital-bed availability and income levels
- Examination of whether healthcare capacity moderates the vaccination–outcome relationship
- Policy-oriented synthesis describing which countries benefit most and why
- Additional visualizations saved to the results folder, reinforcing interpretation

**Additional Deliverables**
- Comprehensive README explaining setup, execution, methodology, and project structure
- Complete API documentation set as required
- Fully containerized execution using Docker
- Jupyter notebook for interactive exploration
- Final script for video walkthrough

---

## 5. Key Features of the Pipeline

###  Automated Data Acquisition  
Downloads OWID compact dataset and stores it locally for reproducibility.

###  Weekly Panel Construction  
Transforms daily data into a country-week panel with aligned time indices.

###  Feature Engineering  
- 3-week lags of cases/deaths/vaccination  
- 3-week rolling averages  
- Confounder extraction and integrity checks  

###  Causal Inference Framework  
Includes:
- OLS vs 2SLS (Instrumental Variables)
- Propensity Score Matching
- Inverse Probability Weighting
- Stratification / blocking  
- Early- vs late-adopter comparisons  
- Heterogeneous treatment effects  

###  Robustness & Sensitivity Analysis  
- Rosenbaum bounds  
- Bootstrap confidence intervals  
- Alternative model specifications  
- Subgroup analysis  

###  Validation  
- Leave-one-country-out CV  
- Temporal out-of-sample testing  
- Placebo tests  
- Positive/negative controls  
- Heteroskedasticity & non-linearity tests  

###  Healthcare System Analysis  
Evaluates how policy effectiveness differs across:
- Bismarck  
- Beveridge  
- National Health Insurance  
- Out-of-Pocket  
- Mixed systems  

###  Full Logging System (Tee Implementation)  
Every printed line including tables, diagnostics, estimates, and validation results is duplicated into `output_results.txt`.  
This ensures the final log matches the full console run exactly.

---

## 6. Outputs Produced

### Text Output  
**`results/output_results.txt`**  
Contains:
- Data summaries  
- Confounder checks  
- All causal estimates  
- Robustness and sensitivity diagnostics  
- Validation results  
- Healthcare system comparisons  

### Generated Plots  
Saved under:
**`results/plots`**  

Examples:
- Rolling average trends  
- Vaccination vs case rate trends  
- Scatterplots for confounders  
- IV diagnostics  
- Propensity score distributions  
- Heterogeneity plots  
- Validation figures  
- Healthcare system comparison charts  

### Processed Data  
Saved as:  
`results/weekly_cleaned.pkl`

---

## 7. Dependencies

See `requirements.txt` for full versions.  
Major libraries include:

```
pandas
numpy
matplotlib
seaborn
statsmodels
linearmodels
causalinference
pycountry
pyarrow
```

---

## 8. API Documentation

This project includes a complete lightweight API layer with:

- **Causal_Inference.API.md** – Detailed specification of both native and wrapper functions  
- **Causal_Inference.API.ipynb** – Demonstration notebook  
- **Causal_Inference.example.md** – A complete end-to-end application using the API layer  

This ensures the pipeline is modular, reusable, and extendable.

---

## 9. Limitations & Considerations

- Observational data cannot guarantee full confounder control.  
- Some IVs may violate exclusion restrictions.  
- Country-level aggregation hides within-country variation.  
- Confounding by waves, variants, or vaccine type remains possible.  

These limitations are documented and discussed in the log output.

---

## 10. Conclusion

This repository provides a fully functioning causal inference engine for analyzing public health interventions using global COVID-19 data.  
It is:

- Reproducible  
- Modular  
- Extensible  
- Well-documented  
- Suitable for academic submission or applied research  

All components required for evaluation including the API documentation, example usage, notebook, plots, logs, and pipeline code are included.

