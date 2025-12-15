# Evaluating the Impact of Dietary Supplement Use on Health Outcomes with EconML (NHANES 2021–2023)

**Course**: MSML610 — Fall 2025  
**Project**: TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes  
**Author(s)**: Karthik Vakada, Sri Akash Kadali  
**Last Updated**: 2025-12-14  

---

## Table of Contents
- [Project Overview and Goals](#project-overview-and-goals)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions (Docker - Recommended)](#setup-instructions-docker---recommended)
  - [Setup Instructions (Local - Optional)](#setup-instructions-local---optional)
- [Usage](#usage)
- [API vs Example Layers](#api-vs-example-layers)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [References](#references)

<!-- tocstop -->

---

## Project Overview and Goals

This project is a **hands-on tutorial** that demonstrates how to use **EconML** to estimate the causal effect of a health intervention using **NHANES (2021–2023)** data.

### Main causal question
> Does **any dietary supplement use** (binary treatment) have a measurable causal effect on:
- **Mean systolic blood pressure** (`sbp_mean`)
- **Fasting plasma glucose** (`fasting_glucose_mg_dl`)?

### Goals
- Build a clean, merged **analysis dataset** from multiple NHANES component CSVs.
- Estimate causal effects using **DRLearner (Double Robust Learner)** from EconML.
- Report:
  - **ATE** (Average Treatment Effect)
  - **Bootstrap CI** for ATE
  - **CATE** (individual-level effects) and simple heterogeneity summaries (age/BMI bins)
- Compare with a transparent baseline:
  - **OLS with robust (HC3) SEs** using statsmodels

---

## Project Structure

This project lives under the MSML610 projects directory:

```text
Projects/
└── TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes/
    ├── README.md
    ├── how_to_run.md
    ├── econml.API.md
    ├── econml.example.md
    ├── econml.API.ipynb
    ├── econml.example.ipynb
    ├── econml.API.py
    ├── econml.example.py
    ├── econml_utils.py
    ├── requirements.txt
    ├── Dockerfile
    ├── docker_name.sh
    ├── docker_build.sh
    ├── docker_bash.sh
    ├── docker_jupyter.sh
    ├── run_jupyter.sh
    ├── install_common_packages.sh
    ├── install_jupyter_extensions.sh
    ├── version.sh
    ├── bashrc
    ├── etc_sudoers
    ├── data/
    ├── figs/
    ├── tables/
    ├── tmp_build/
    └── changelog.txt
````

---

## How It Works

### High-level pipeline

```mermaid
flowchart TD
    A[Start] --> B[Load cleaned NHANES component CSVs from data/]
    B --> C[Merge into single analysis DataFrame by respondent id (SEQN)]
    C --> D[Derive variables: sbp_mean, treatment_supplement, covariates]
    D --> E[Create (Y, T, X) via econml_utils.get_y_t_x()]
    E --> F[Fit DRLearner (regression + propensity)]
    F --> G[Compute ATE + bootstrap CI]
    F --> H[Compute individual effects (CATE / tau_hat)]
    H --> I[Summarize heterogeneity by age/BMI quartiles]
    D --> J[Run OLS baseline with HC3 SE]
    G --> K[Return results as plain dict/DataFrame]
    I --> K
    J --> K
    K --> L[End]
```

### “Schema” of the structured inputs (what gets merged)

The analysis dataset is built by joining multiple NHANES component files on respondent id (`SEQN`). Typical components used here include:

* `DEMO_*` (demographics: age/sex)
* `BMX_*` (body measures: BMI, weight, waist)
* `BPXO_*` (blood pressure readings → `sbp_mean`)
* `GLU_*` (fasting glucose)
* `TCHOL_*`, `HDL_*`, `TRIGLY_*` (lipids)
* `HSCRP_*` (hs-CRP)
* `DSQTOT_*` (dietary supplement use → `treatment_supplement`)

Exact file names/columns are implemented in `econml_utils.py`.

---

## Getting Started

### Prerequisites

* **Docker Desktop** installed and running
* Enough disk space (image is ~1–2GB once built)
* macOS is supported (Apple Silicon is fine)

### Setup Instructions (Docker - Recommended)

From the project folder:

```bash
cd /Users/karthikvakada/src/umd_classes1/class_project/MSML610/Fall2025/Projects/TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
```

Make scripts executable (one-time):

```bash
chmod +x docker_*.sh run_jupyter.sh install_*.sh version.sh
```

Build the Docker image:

```bash
./docker_build.sh
```

Start Jupyter (inside container):

```bash
./docker_jupyter.sh
```

Expected output looks like:

```text
Starting Jupyter in /curr_dir on port 8888 ...
You should be able to open it at: http://localhost:8888
```

Open the printed URL in your browser.

### Setup Instructions (Local - Optional)

Docker is the expected path for grading. If you still want local execution:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

---

## Usage

### 1) Run the API notebook (minimal, “how to call functions”)

Open and run:

* `econml.API.ipynb`

This notebook demonstrates:

* `build_analysis_df()` and `get_y_t_x()` from `econml_utils.py`
* DRLearner experiments and OLS baseline from `econml.API.py`

### 2) Run the example notebook (end-to-end tutorial)

Open and run:

* `econml.example.ipynb`

This notebook is the recommended “start here” path:

* Builds the analysis dataset
* Runs DRLearner for SBP + glucose
* Shows ATE/CATE summaries + compares to OLS

### 3) (Optional) Run the script version

Start a shell in the container:

```bash
./docker_bash.sh
```

Then run:

```bash
python econml.example.py
```

---

## API vs Example Layers

### API / Core utilities (stable “contract”)

* `econml_utils.py`

  * `build_analysis_df()`: loads + merges NHANES component CSVs and creates derived variables
  * `get_y_t_x(df, outcome_col, treatment_col)`: returns `(Y, T, X, covariates)` consistently

* `econml.API.py`

  * `run_sbp_supplement_experiment(...)`
  * `run_glucose_supplement_experiment(...)`
  * `run_ols_for_outcome(...)`

These functions return **plain Python objects / DataFrames** so notebooks stay clean and reusable.

### Example / Tutorial layer (student-facing)

* `econml.API.ipynb`: API demonstration
* `econml.example.ipynb`: end-to-end walkthrough
* `econml.example.py`: script equivalent for quick regression testing

---

## Reproducibility Notes

* Notebooks are designed to be run top-to-bottom using **Restart & Run All**
* DRLearner uses a fixed `random_state` by default
* ATE confidence intervals are computed with a simple nonparametric bootstrap
* The project expects cleaned `*_meaningful*.csv` inputs to exist in `data/`

---

## Troubleshooting

### 1) “Jupyter command `jupyter-notebook` not found”

Some environments include JupyterLab but not the classic `notebook` package.
Use JupyterLab:

* Inside container: make sure `run_jupyter.sh` launches `jupyter lab` (not `jupyter-notebook`)
* If you’re running manually in the container, use:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''
```

### 2) “exec format error” when running `run_jupyter.sh`

This usually means `run_jupyter.sh` has Windows CRLF line endings or a broken shebang.
Fix line endings on macOS:

```bash
sed -i '' $'s/\r$//' run_jupyter.sh
chmod +x run_jupyter.sh
```

Also ensure the first line of `run_jupyter.sh` is:

```bash
#!/usr/bin/env bash
```

### 3) Port 8888 already in use

Either stop the process using it, or change the port mapping in `docker_jupyter.sh` (e.g., host 8889 → container 8888).

### 4) Apple Silicon note

If you ever hit a wheel/build issue on Apple Silicon, rebuild forcing amd64:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
./docker_build.sh
```

---

## References

* EconML documentation: [https://econml.azurewebsites.net/](https://econml.azurewebsites.net/)
* NHANES Continuous (2021–2023): [https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)
* scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
* statsmodels: [https://www.statsmodels.org/](https://www.statsmodels.org/)

```


