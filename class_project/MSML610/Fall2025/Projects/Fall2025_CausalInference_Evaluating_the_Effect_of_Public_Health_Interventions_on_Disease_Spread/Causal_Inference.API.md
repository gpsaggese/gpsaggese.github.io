# COVID-19 Causal Analysis API Documentation  
Tool: Public Health Intervention Impact Analyzer  
Module Location: `src/`
Author: Anto Delin Xavier  

---

## Overview

This document explains the internal API used by our analysis system to:

- Load COVID-19 epidemiological data
- Construct weekly intervention panels
- Engineer causal analysis features
- Estimate causal effects (regression + IV)
- Validate model robustness
- Produce policy-relevant insights

The system is composed of modular Python files in `src/`, and orchestrated through a wrapper in `Causal_Inference.example.py`.

---

## Module: `data_loader.py`

### `download_owid_data() → pd.DataFrame`
Downloads the OWID Compact COVID-19 dataset.

**Returns:**  
Daily country-level data including:
- Case counts
- Deaths
- Vaccination progress
- Population controls

---

## Module: `preprocess.py`

### `clean_data_minimal(df) → pd.DataFrame`
- Filters valid ISO3 country records  
- Standardizes country code column  
- Parses date fields  

### `build_weekly_panel(df) → pd.DataFrame`
Creates weekly aggregated features:
- `week_start` (ISO week)
- Weekly cases & deaths per 100k
- Interpolated vaccination percentage

### `final_clean(df) → pd.DataFrame`
Ensures required variables are present and removes unusable entries.

---

## Module: `feature_eng.py`

### `add_features(df)`
Calls the functions below:

#### `create_lagged_features(df, lag_weeks=3)`
Adds **3-week lagged** features:
- `vac_pct_lag`
- `cases_per_100k_lag`
- `deaths_per_100k_lag`

#### `create_rolling_features(df, window=3)`
Adds **3-week rolling means** for outcomes & vaccination

---

## Module: `causal_analysis.py`

This module contains the **entire analysis pipeline from Task 2.1 to Bonus**.

### Task 2 — Exploratory Causal Structure

| Subtask | Function | Purpose |
|--------|----------|---------|
| 2.1 | `plot_lagged_relationship(df, plots_dir, log)` | Temporal ordering & visual checks |
| 2.2 | `plot_rolling_relationships(df, plots_dir, log)` | Correlations & curve visualization |

---

### Task 3 — Causal Effect Estimation

| Subtask | Function | Method |
|--------|----------|--------|
| 3.1 | `estimate_treatment_effect(df, log)` | Regression with confounders |
| 3.2 | `estimate_iv_effect(df, log)` | Instrumental variables (2SLS) |
| 3.3 | `estimate_binary_treatment_effect(df, log)` | Discrete threshold treatment |

---

### Task 4 — Heterogeneous Treatment Effects

| Subtask | Function | Output |
|--------|----------|--------|
| 4.1 | `estimate_subgroup_effects(df, log)` | ATE by continent & system factors |
| 4.2 | `estimate_causal_forest(df, plots_dir, log)` | Machine-learning heterogeneity |
| 4.3 | `plot_hte_results(results, plots_dir, log)` | Visual subgroup comparison |
| 4.4 | `interpret_hte_results(results, log)` | Policy-relevant interpretation |
| 4.5 | `run_model_validation(df, plots_dir, log)` | Diagnostics, cross-validation |

---

### Bonus — Healthcare System Moderation

| Function | Purpose |
|---------|---------|
| `healthcare_moderator_analysis(df, plots_dir, log)` | Compare outcomes across system types |

---

## Module: `utils.py`

### `setup_logging(log_file_path)`
Routes output to both:
- Console
- Results file (`results/output_results.txt`)

### `log_print(message)`
Central print function for consistent text formatting.

---

## Wrapper: `Causal_Inference.example.py`

### `run_analysis_workflow()`
Executes all tasks in **the same order as the notebook**:

1. Load raw data
2. Build weekly panel
3. Feature engineering
4. Tasks 2–4
5. Validation
6. Bonus study
7. Save:
   - Plots to `/results/plots/`
   - Text output to `/results/output_results.txt`

---

## Summary

This API provides a **reproducible causal inference framework**, enabling:
- Modular experiment refinement
- Automated full-pipeline execution
- Academic-grade results mirroring the notebook

The notebook (`Causal_Inference.API.ipynb`) demonstrates usage of both the **native modules** and the **wrapper interface**.

