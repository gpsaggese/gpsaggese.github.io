- [Project Overview](#project-overview)
- [Project Files](#project-files)
- [Setup and Dependencies](#setup-and-dependencies)
  * [Building and Running the Docker Container](#building-and-running-the-docker-container)
    + [Environment Setup](#environment-setup)
- [How to Use This Tutorial](#how-to-use-this-tutorial)
- [Data and Experiment Design](#data-and-experiment-design)
- [API vs Example Layers](#api-vs-example-layers)
- [Reproducibility Notes](#reproducibility-notes)

<!-- tocstop -->

# Project Overview

- **Course:** MSML610 – Fall 2025  
- **Project ID:** TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes  
- **Author:** Karthik Vakada  
- **Date:** 2025-03-15  

This project is a **hands-on tutorial** showing how to use **EconML** to estimate the causal effect
of a health intervention using **NHANES (2021–2023)** data.

We treat **“any dietary supplement use”** as the binary treatment and study its impact on:

- Mean systolic blood pressure (**`sbp_mean`**)
- Fasting plasma glucose (**`fasting_glucose_mg_dl`**)

The repository is structured as a **tool-style tutorial** with:

- A **lightweight API layer** (`econml_utils.py` + `econml.API.py`)
- A **reference API notebook** (`econml.API.ipynb`)
- A **story-style example notebook & script** (`econml.example.*`)
- Docker scripts so another student can reproduce everything in ~60 minutes

---

# Project Files

This project lives under:

```text
class_project/MSML610/Fall2025/Projects/
└── TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes/
````

Core files in this folder:

* **Documentation / tutorial**

  * `README.md`
    This file. High-level overview, structure, and how to run everything.
  * `how_to_run.md`
    Shorter, task-focused “run this project” guide (duplicate of the key steps here).
  * `econml.API.md`
    Markdown documentation for the **API layer**: functions, parameters, return values, and design choices.
  * `econml.example.md`
    Narrative tutorial walking through the full example: building the dataset, running EconML, and interpreting results.

* **Python modules (API + utils)**

  * `econml_utils.py`
    Core data utilities for this project:

    * Loading pre-cleaned NHANES CSVs from `data/`
    * Merging them into a single analysis DataFrame
    * Defining the treatment and covariates
    * Returning `(Y, T, X)` for EconML / OLS
  * `econml.API.py`
    High-level API wrapper that exposes:

    * `run_sbp_supplement_experiment(...)`
    * `run_glucose_supplement_experiment(...)`
    * `run_ols_for_outcome(...)`
      This file is the “contract” layer for the tutorial.

* **Notebooks**

  * `econml.API.ipynb`

    * Demonstrates how to import and use the functions from `econml_utils.py` and `econml.API.py`.
    * Focuses on **API usage**, not storytelling.
  * `econml.example.ipynb`

    * Main end-to-end tutorial notebook.
    * Designed for a student who wants to learn the workflow in ~60 minutes.

* **Scripts**

  * `econml.example.py`

    * Script version of the example pipeline.
    * Useful for non-notebook execution and quick regression tests.

* **Data**

  * `data/`
    Pre-cleaned NHANES component files (each in “*_meaningful.csv” format), for example:

    * `BMX_L_meaningful*.csv` – body measures (BMI, weight, waist circumference)
    * `BPXO_L_meaningful*.csv` – blood pressure readings
    * `TCHOL_L_meaningful*.csv` – total cholesterol
    * `HDL_L_meaningful*.csv` – HDL cholesterol
    * `TRIGLY_L_meaningful*.csv` – triglycerides
    * `GLU_L_meaningful*.csv` – fasting glucose
    * `HSCRP_L_meaningful*.csv` – hs-CRP
    * `DSQTOT_L_meaningful*.csv` – dietary supplements (used to define treatment)
    * `DEMO_L_meaningful*.csv` – demographics (age, sex, etc.)

* **Docker / environment**

  * `Dockerfile`
    Base image and dependencies for MSML610 “thin” environment.
  * `requirements.txt`
    Python dependencies for this project (EconML, scikit-learn, pandas, numpy, etc.).
  * `docker_name.sh`
    Central definition of image and container names.
  * `docker_build.sh`
    Builds the Docker image for this project.
  * `docker_bash.sh`
    Starts an interactive bash shell in the container.
  * `docker_jupyter.sh`
    Starts a container and launches Jupyter (mapped to localhost).
  * `run_jupyter.sh`
    Script executed inside the container to run Jupyter.

* **Misc**

  * `Data_Preparation_Sri.ipynb`
    Auxiliary notebook that was used to create the cleaned `*_meaningful.csv` files.
  * `docker_build.log`, `docker_build.version.log`
    Logs from Docker builds.
  * `changelog.txt`
    Optional file capturing high-level changes over time.

> The project also relies on the standard MSML610 devops/thin-client tooling
> from the class repository; those shared files are not duplicated here.

---

# Setup and Dependencies

This project is intended to run inside a **Docker container** using the provided
scripts. You do **not** need external APIs or internet access once the image is built.

## Building and Running the Docker Container

From the **project folder**:

```bash
cd class_project/MSML610/Fall2025/Projects/TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
```

### Build the Docker image (first time)

```bash
bash docker_build.sh
```

* Uses the local `Dockerfile`
* Builds an image (name/tag managed by `docker_name.sh`)
* Installs dependencies from `requirements.txt`

### Launch Jupyter inside the container

```bash
bash docker_jupyter.sh
```

This will:

1. Start a container from the MSML610 image.
2. Mount the project directory into `/curr_dir` inside the container.
3. Run `run_jupyter.sh`, which launches Jupyter on port 8888.

The terminal will print a URL similar to:

```text
http://127.0.0.1:8888/?token=...
```

Open that URL in your browser to access the notebooks.

### Environment Setup

In normal use, **no extra setup** is required inside Docker. If you ever need to
reinstall dependencies manually:

```bash
# Inside the running container
pip install -r requirements.txt
```

If you want to run **without Docker** (not required for the class):

```bash
python -m venv .venv
source .venv/bin/activate   # on macOS / Linux
# .venv\Scripts\activate    # on Windows

pip install -r requirements.txt
```

Then run the notebooks/scripts with your local Python.

---

# How to Use This Tutorial

Once Jupyter is running (via `docker_jupyter.sh`):

1. Open **`econml.example.ipynb`** (recommended first)

   * Run “Restart & Run All”.
   * This notebook:

     * Builds the merged NHANES analysis DataFrame.
     * Defines treatment (`treatment_supplement`) and outcomes.
     * Calls the API functions from `econml.API.py`.
     * Shows and interprets ATE/CATE estimates and OLS baseline.

2. Open **`econml.API.ipynb`** (reference)

   * Demonstrates:

     * `build_analysis_df()` and `get_y_t_x()` from `econml_utils.py`.
     * `run_sbp_supplement_experiment()`, `run_glucose_supplement_experiment()`,
       and `run_ols_for_outcome()` from `econml.API.py`.
   * Focuses on the **programming interface**, not on detailed narrative.

3. (Optional) Run the script version:

   ```bash
   bash docker_bash.sh          # open a shell inside the container
   python econml.example.py     # run the full example pipeline
   ```

The script mirrors the main logic of `econml.example.ipynb`: build data → run
EconML → print key results.

---

# Data and Experiment Design

* **Source:** NHANES continuous 2021–2023 (pre-processed into `*_meaningful.csv` files under `data/`).
* **Unit of analysis:** Respondent (joined across multiple NHANES components).

### Treatment

* Column: `treatment_supplement`
* Definition:

  * `1` if the respondent reported **any dietary supplement use**
  * `0` otherwise

### Outcomes

* `sbp_mean`
  Mean systolic blood pressure across 3 oscillometric readings.

* `fasting_glucose_mg_dl`
  Fasting plasma glucose (mg/dL).

### Covariates

A common set of covariates is used for both EconML and OLS models, including:

* `age_years`
* `sex`
* `body_mass_index_kg_m2`
* `weight_kg`
* `waist_circumference_cm`
* `total_cholesterol_mg_dl`
* `direct_hdl_cholesterol_mg_dl`
* `LBXTLG` (triglycerides)
* `fasting_glucose_mg_dl` (as covariate when outcome is not glucose itself)
* `hs_c_reactive_protein_mg_l`

The exact feature set and any filtering steps are documented in
`econml_utils.py` and explained in `econml.API.md`.

---

# API vs Example Layers

The structure follows the course guideline of **separating a stable API layer**
from a runnable example implementation.

### API / Core utilities

* **`econml_utils.py`**

  * `build_analysis_df()`

    * Loads all required `*_meaningful.csv` files from `data/`
    * Merges them on respondent ID
    * Creates derived variables (e.g., mean SBP, treatment flag)
  * `get_y_t_x(analysis_df, outcome_col, treatment_col)`

    * Returns `(Y, T, X, covariate_names)` for a given outcome and treatment.

* **`econml.API.py`**

  * `run_sbp_supplement_experiment(random_state=42)`

    * Uses **EconML’s DRLearner** to estimate:

      * ATE of supplement use on SBP
      * CATEs and simple heterogeneity summaries
  * `run_glucose_supplement_experiment(random_state=42)`

    * Same as above but for fasting glucose.
  * `run_ols_for_outcome(outcome_col, treatment_col="treatment_supplement")`

    * Runs a standard OLS regression with the same covariates.
    * Returns the treatment coefficient as a baseline estimate.

These functions are the **“stable contract”**: users of the tutorial should be
able to import and call them without caring about internal data joins or model
configuration.

### Example / Tutorial layer

* **`econml.API.ipynb`**

  * Minimal cells showing how to import and call the API functions.
  * Good starting point if you want to use this code in your own project.

* **`econml.example.ipynb`**

  * Full, student-facing tutorial:

    * Context + motivation
    * Data exploration (brief)
    * EconML DRLearner runs for SBP and glucose
    * ATE/CATE interpretation
    * Comparison with OLS

* **`econml.example.py`**

  * Script version of the same flow, returning/printing summary stats.

---

# Reproducibility Notes

* All notebooks are designed to be **run top-to-bottom** with:

  * Jupyter: *Kernel → Restart & Run All*
* Randomness:

  * Models are initialized with fixed `random_state` values (e.g., `42`) where relevant.
* External dependencies:

  * No external APIs or internet access are required after the Docker image is built.
* Data:

  * All required NHANES component CSVs are expected to be present in `data/`.
  * If you change file names or paths, update `econml_utils.py` accordingly.

If all steps are followed, another student should be able to clone the class repo,
build the image, run the notebooks, and reproduce the reported causal estimates
within about an hour.
