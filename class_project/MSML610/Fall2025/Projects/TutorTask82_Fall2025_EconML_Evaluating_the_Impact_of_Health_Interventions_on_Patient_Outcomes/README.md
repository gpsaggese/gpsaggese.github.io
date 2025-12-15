# Evaluating the Impact of Dietary Supplement Use on Health Outcomes with EconML (NHANES 2021–2023)

**Course**: MSML610 — Fall 2025
**Project**: TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
**Author(s)**: Karthik Vakada, Sri Akash Kadali
**Last Updated**: 2025-12-14

---

## Table of Contents

* [Project Overview and Goals](#project-overview-and-goals)
* [Project Structure](#project-structure)
* [How It Works](#how-it-works)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Setup Instructions (Docker – Recommended)](#setup-instructions-docker--recommended)
  * [Setup Instructions (Local – Optional)](#setup-instructions-local--optional)
* [Usage](#usage)
* [API vs Example Layers](#api-vs-example-layers)
* [Reproducibility Notes](#reproducibility-notes)
* [Troubleshooting](#troubleshooting)
* [References](#references)

---

## Project Overview and Goals

This project is a **hands-on causal inference tutorial** demonstrating how to estimate treatment effects using **EconML** on real-world observational health data from **NHANES (2021–2023)**.

The focus is not prediction. The focus is **causal estimation under confounding** using modern double-robust methods.

---

### Main causal question

> Does **any dietary supplement use** (binary treatment) have a causal effect on:

* **Mean systolic blood pressure** (`sbp_mean`)
* **Fasting plasma glucose** (`fasting_glucose_mg_dl`)?

---

### Project goals

* Build a **clean, merged, analysis-ready dataset** from multiple NHANES component files
* Estimate causal effects using **DRLearner (Double Robust Learner)**
* Report:

  * **ATE** (Average Treatment Effect)
  * **Bootstrap confidence intervals**
  * **CATE** (individual-level treatment effects) with basic heterogeneity analysis
* Compare EconML estimates against a transparent baseline:

  * **OLS with HC3 robust standard errors** using `statsmodels`

---

## Project Structure

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
```

---

## How It Works

### High-level pipeline

```mermaid
flowchart TD
    A[Start] --> B[Load cleaned NHANES CSVs from data/]
    B --> C[Merge by respondent ID SEQN]
    C --> D[Derive outcomes, treatment, covariates]
    D --> E[Create Y, T, X matrices]
    E --> F[Fit DRLearner]
    F --> G[Estimate ATE + CI]
    F --> H[Estimate CATE]
    H --> I[Heterogeneity summaries]
    D --> J[OLS baseline with HC3 SE]
    G --> K[Collect results]
    I --> K
    J --> K
    K --> L[End]
```

---

### NHANES components used

All datasets are merged using the respondent identifier `SEQN`.

* `DEMO_*` — demographics
* `BMX_*` — body measures (BMI, weight)
* `BPXO_*` — blood pressure readings → `sbp_mean`
* `GLU_*` — fasting plasma glucose
* `TCHOL_*`, `HDL_*`, `TRIGLY_*` — lipid profile
* `HSCRP_*` — high-sensitivity C-reactive protein
* `DSQTOT_*` — dietary supplement use → treatment indicator

All cleaning, feature engineering, and consistency checks are implemented in **`econml_utils.py`**.

---

## Getting Started

## Prerequisites

### All platforms

* Docker Desktop installed and running
* Approximately **1–2 GB** of free disk space

### Windows-specific

* Docker Desktop with **WSL 2 backend enabled**
* Git Bash or WSL recommended for running `.sh` scripts
  (Native PowerShell is not ideal for shell scripts)

---

## Setup Instructions (Docker – Recommended)

Docker is the **official and graded execution path**.
These steps work on **macOS, Linux, Windows (WSL/Git Bash)**.

---

### Step 1: Navigate to the project directory

#### macOS / Linux

```bash
cd /Users/karthikvakada/src/umd_classes1/class_project/MSML610/Fall2025/Projects/TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
```

#### Windows (PowerShell)

```powershell
cd C:\Users\karthikvakada\src\umd_classes1\class_project\MSML610\Fall2025\Projects\TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
```

If using WSL, use Linux-style paths inside the WSL environment.

---

### Step 2: Make scripts executable (one-time)

#### macOS / Linux / WSL

```bash
chmod +x docker_*.sh run_jupyter.sh install_*.sh version.sh
```

#### Windows

* Use **Git Bash or WSL**
* No `chmod` needed in native PowerShell

---

### Step 3: Build the Docker image

```bash
./docker_build.sh
```

(On Windows, run the same command from Git Bash or WSL.)

---

### Step 4: Launch Jupyter

```bash
./docker_jupyter.sh
```

Expected output:

```text
Starting Jupyter in /curr_dir on port 8888 ...
You should be able to open it at: http://localhost:8888
```

Open the URL in your browser.

---

## Setup Instructions (Local – Optional)

Not recommended for grading. Use Docker unless explicitly required.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

### Windows

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

---

## Usage

### Option 1: API notebook (minimal)

Run:

* `econml.API.ipynb`

Demonstrates:

* Dataset construction
* DRLearner calls
* OLS baseline comparison

---

### Option 2: Example notebook (recommended)

Run:

* `econml.example.ipynb`

Covers:

* Full end-to-end workflow
* SBP and glucose experiments
* ATE, CATE, and model comparisons

---

### Option 3: Script execution

```bash
./docker_bash.sh
python econml.example.py
```

---

## API vs Example Layers

### Core API (stable, reusable)

**`econml_utils.py`**

* `build_analysis_df()`
* `get_y_t_x()`

**`econml.API.py`**

* `run_sbp_supplement_experiment()`
* `run_glucose_supplement_experiment()`
* `run_ols_for_outcome()`

All functions return plain Python objects or pandas DataFrames.

---

### Example layer (educational)

* `econml.API.ipynb`
* `econml.example.ipynb`
* `econml.example.py`

These scripts orchestrate experiments and visualization but do not introduce new logic.

---

## Reproducibility Notes

* All notebooks support **Restart & Run All**
* Random seeds are fixed where applicable
* Bootstrap is used for ATE confidence intervals
* Requires cleaned `*_meaningful*.csv` files in `data/`

---

## Troubleshooting

### Jupyter not reachable

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
```

---

### `exec format error` on Windows

Convert line endings:

```bash
sed -i 's/\r$//' run_jupyter.sh
```

Ensure the first line is:

```bash
#!/usr/bin/env bash
```

---

### Port 8888 already in use

Edit `docker_jupyter.sh` and change the host port mapping.

---

### Apple Silicon (M1/M2/M3)

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
./docker_build.sh
```

---

## References

* EconML: [https://econml.azurewebsites.net/](https://econml.azurewebsites.net/)
* NHANES 2021–2023: [https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023)
* scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
* statsmodels: [https://www.statsmodels.org/](https://www.statsmodels.org/)
