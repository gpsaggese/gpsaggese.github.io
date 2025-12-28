

# Causal_Inference.example.md

# Example Application Using the Causal Analysis API Layer

This document provides a complete, end‑to‑end example of how to build an application using the lightweight API wrapper created for the project: **Evaluating the Effect of Public Health Interventions on Disease Spread**.

The example demonstrates how a typical Python script—or another system—can interact with the API layer without needing to know any internal implementation details.

---

## 1. Overview of the Example Application

The example performs the following tasks using only the wrapper API functions:

1. Load raw OWID COVID‑19 data.
2. Clean and preprocess the dataset.
3. Construct weekly country‑level panel data.
4. Engineer lagged and rolling features.
5. Run a full causal analysis workflow (Tasks 2–4 + Validation + Policy + Bonus analyses).
6. Save all results and plots.

This mirrors the entire project pipeline but exposes a clean external interface.

---

## 2. Directory Structure Expected by the Example

```
project_root/
│
├── main.py
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_eng.py
│   ├── causal_analysis.py
│   └── utils.py
│
├── results/
│   ├── output_results.txt
│   └── plots/
└── data/
    ├── raw/
    └── processed/
```

---

## 3. Example Usage Code

Below is the complete minimal script illustrating how an external application interacts with the API wrapper.

```python
from src.data_loader import download_owid_data
from src.preprocess import clean_data_minimal, build_weekly_panel, final_clean
from src.feature_eng import add_features
from src.causal_analysis import run_analysis_workflow
from src.utils import setup_logging, log_print

import os

# ---------------------------------------------------------------------
# 1. Setup output directories and logging
# ---------------------------------------------------------------------
PROJECT_DIR = "/path/to/project_root"  # modify as needed
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
LOG_FILE = os.path.join(RESULTS_DIR, "output_results.txt")

setup_logging(RESULTS_DIR, LOG_FILE)
log_print("Starting Example Application Using the API Layer")

# ---------------------------------------------------------------------
# 2. Task 1 — Data Acquisition
# ---------------------------------------------------------------------
raw_df = download_owid_data()
log_print(f"Loaded raw dataset: {raw_df.shape[0]:,} rows")

# ---------------------------------------------------------------------
# 3. Data Cleaning & Weekly Panel Construction
# ---------------------------------------------------------------------
clean_df = clean_data_minimal(raw_df)
weekly_df = build_weekly_panel(clean_df)
log_print(f"Weekly panel constructed: {weekly_df.shape[0]:,} rows")

# ---------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------
feature_df = add_features(weekly_df)
final_df = final_clean(feature_df)
log_print(f"Final analysis dataset: {final_df.shape[0]:,} rows")

# ---------------------------------------------------------------------
# 5. Full Causal Analysis Pipeline
# ---------------------------------------------------------------------
run_analysis_workflow(final_df, log_print, PLOTS_DIR)
log_print("Full analysis complete. Outputs saved.")
```

---

## 4. Explanation of How This Example Uses the Wrapper Layer

### The example script:

* **Does not manipulate raw DataFrame structures manually.**
* **Does not call internal helper functions directly.**
* **Only interacts with public wrapper functions**, such as:

  * `download_owid_data()`
  * `clean_data_minimal()`
  * `build_weekly_panel()`
  * `add_features()`
  * `final_clean()`
  * `run_analysis_workflow()`
  * `log_print()`

These form the stable API contract the project guarantees.

---

## 5. Expected Outputs

When executed, the example application will produce:

### Text Output

Saved automatically to:

```
results/output_results.txt
```

This file includes:

* Task 1 results
* All Task 2–4 causal analysis outputs
* Validation diagnostics
* Policy recommendations
* Bonus healthcare‑system results

### Visualization Outputs

Saved to:

```
results/plots/
```

Including:

* ATE comparisons
* Lagged‑effect scatterplots
* Matching diagnostics
* IV regression plots
* Heterogeneous‑effects bar charts
* Validation plots
* Healthcare‑system comparison visuals

---

## 6. Running the Example

After saving the example script as **main.py**, run:

```
python main.py
```

All analysis tasks and subtasks will execute sequentially.

---

## 7. Conclusion

This example demonstrates how an external application can use the project’s API layer to:

* Load data
* Preprocess and engineer features
* Run a complex causal analysis pipeline
* Save results and visualizations

Without knowing any internal implementation details, This ensures reproducibility, maintainability, and ease of integration with other systems.
