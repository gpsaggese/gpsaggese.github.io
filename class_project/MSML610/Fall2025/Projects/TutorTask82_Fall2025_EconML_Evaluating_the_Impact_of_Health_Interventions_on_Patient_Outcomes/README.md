
# EconML – Evaluating the Impact of Health Interventions on Patient Outcomes  
MSML610 – Fall 2025

**Project folder:**  
`TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes`

This project uses **NHANES 2021–2023** data and **EconML’s DRLearner** to estimate the effect of
**any dietary supplement use** on two health outcomes:

- Mean systolic blood pressure (`sbp_mean`)
- Fasting plasma glucose (`fasting_glucose_mg_dl`)

The goal is to provide a **clear API + example tutorial** that another student can follow in about
60 minutes.

---

## Project layout

```text
TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes/
├── econml_utils.py          # Core data utilities (build_analysis_df, get_y_t_x)
├── econml.API.py            # High-level API wrapper around EconML + OLS
├── econml.API.md            # Markdown documentation for the API layer
├── econml.API.ipynb         # Notebook explaining and demoing the API functions
├── econml.example.py        # Script version of the example tutorial
├── econml.example.md        # Narrative markdown tutorial
├── econml.example.ipynb     # Main end-to-end example notebook (student-facing)
├── data/
│   ├── BMX_L_meaningful*.csv      # Body measures
│   ├── BPXO_L_meaningful*.csv     # Blood pressure readings
│   ├── TCHOL_L_meaningful*.csv    # Total cholesterol
│   ├── HDL_L_meaningful*.csv      # HDL cholesterol
│   ├── TRIGLY_L_meaningful*.csv   # Triglycerides
│   ├── GLU_L_meaningful*.csv      # Fasting glucose
│   ├── HSCRP_L_meaningful*.csv    # hs-CRP
│   ├── DSQTOT_L_meaningful*.csv   # Dietary supplements (treatment)
│   └── DEMO_L_meaningful*.csv     # Demographics (age, sex, etc.)
├── Dockerfile                # Image used for MSML610 projects
├── requirements.txt          # Python dependencies (econml, sklearn, pandas, etc.)
├── docker_build.sh           # Build the Docker image
├── docker_name.sh            # Common image/container name variables
├── docker_bash.sh            # Start an interactive shell inside the container
├── docker_jupyter.sh         # Launch Jupyter inside the container
├── run_jupyter.sh            # Script called inside the container to run Jupyter
├── how_to_run.md             # Short “how to run everything” guide
├── README.md                 # This file
└── changelog.txt             # Optional project history
````

---

## Quick start (using Docker)

These steps assume you are in the project folder:

```bash
cd TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
```

### 1. Build the Docker image (first time)

```bash
bash docker_build.sh
```

This builds the image defined in `Dockerfile`. The image name is managed by
`docker_name.sh` and is typically:

* Repository: `umd_msml610`
* Image: `umd_msml610_image`

### 2. Launch Jupyter inside the container

```bash
bash docker_jupyter.sh
```

This will:

1. Start a container using the MSML610 image.
2. Mount the project directory into `/curr_dir` inside the container.
3. Run `run_jupyter.sh`, which launches Jupyter Notebook/Lab on port 8888.

Then open the URL shown in the terminal (usually `http://127.0.0.1:8888`) in your browser.

### 3. Recommended notebooks to run

Inside Jupyter, start with:

1. **`econml.API.ipynb`**

   * Shows how to import:

     * `build_analysis_df`, `get_y_t_x` from `econml_utils.py`
     * `run_sbp_supplement_experiment`, `run_glucose_supplement_experiment`,
       `run_ols_for_outcome` from `econml.API.py`
   * Prints ATEs and shows what each API function returns.
   * Designed as a **reference notebook** for the programming interface.

2. **`econml.example.ipynb`**

   * The main tutorial notebook (student-facing).
   * Walks through:

     1. Building the merged NHANES dataset
     2. Defining treatment and outcomes
     3. Running DRLearner for SBP and fasting glucose
     4. Exploring CATEs and heterogeneity by BMI
     5. Comparing EconML ATE vs a traditional OLS regression (**bonus task**)

If you prefer a pure script, you can also run:

```bash
python econml.example.py
```

inside the container. This mirrors the main steps from `econml.example.ipynb`.

---

## Data and treatment definition

* **Source:** NHANES 2021–2023 continuous survey
  (cleaned CSVs are already placed in `data/`).

* **Treatment** (`treatment_supplement`):

  * 1 if the respondent reported **any dietary supplement use**
  * 0 otherwise

* **Outcomes:**

  * `sbp_mean` — mean systolic BP (3 oscillometric readings)
  * `fasting_glucose_mg_dl` — fasting plasma glucose

* **Covariates (used for both EconML and OLS):**

  ```text
  age_years
  sex
  body_mass_index_kg_m2
  weight_kg
  waist_circumference_cm
  total_cholesterol_mg_dl
  direct_hdl_cholesterol_mg_dl
  LBXTLG                       # triglycerides
  fasting_glucose_mg_dl
  hs_c_reactive_protein_mg_l
  ```

---

## API vs example layer

* **API layer**

  * `econml_utils.py`

    * `build_analysis_df()` – merges all cleaned NHANES components
    * `get_y_t_x(analysis_df, outcome_col, treatment_col)` – returns `Y`, `T`, `X`, and covariate names
  * `econml.API.py`

    * `run_sbp_supplement_experiment(random_state=42)` – DRLearner for SBP
    * `run_glucose_supplement_experiment(random_state=42)` – DRLearner for fasting glucose
    * `run_ols_for_outcome(outcome_col, treatment_col="treatment_supplement")` – OLS baseline

* **Example layer**

  * `econml.API.ipynb` – documents the API and shows direct calls.
  * `econml.example.ipynb` – story-style notebook used as the main tutorial.
  * `econml.example.md` – markdown version of the tutorial.
  * `econml.example.py` – script form of the example.

This separation makes it easy for other students to re-use the API in their own
notebooks without touching the internals.

---

## Results (high-level)

* **SBP (`sbp_mean`)**

  * EconML DRLearner ATE ≈ **–2 mmHg**
  * OLS treatment coefficient ≈ **–1.98 mmHg**
  * Interpretation: supplement users have slightly lower systolic BP, on average, after adjustment.

* **Fasting glucose (`fasting_glucose_mg_dl`)**

  * EconML ATE ≈ **0**
  * OLS treatment coefficient ≈ **0**
  * Interpretation: no meaningful average effect on fasting plasma glucose.

* **Heterogeneity**

  * For SBP, the mean CATE across BMI quartiles is close to the overall ATE.
  * No strong heterogeneity by BMI is observed.
  * For fasting glucose, the CATEs and BMI-bin effects are essentially zero.

These points are explained in more detail inside `econml.example.ipynb`.

````

---

## 2️⃣ New `how_to_run.md` (replace the whole file with this)

```markdown
# How to run this project

This project is designed to be run **inside the MSML610 Docker image**.  
All commands below assume you are in the project folder:

```bash
cd TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes
````

---

## 1. Build the Docker image (first time only)

```bash
bash docker_build.sh
```

This uses the provided `Dockerfile` and the MSML610 base image to create a local
image (name managed by `docker_name.sh`).

---

## 2. Start Jupyter inside the container

```bash
bash docker_jupyter.sh
```

What this does:

1. Starts a container from the MSML610 image.
2. Mounts the current project directory into `/curr_dir` inside the container.
3. Runs `run_jupyter.sh`, which launches Jupyter Notebook/Lab on port 8888.

Open the printed URL (usually `http://127.0.0.1:8888`) in your browser.

Inside Jupyter you should see:

* `econml.API.ipynb`
* `econml.example.ipynb`
* the Python modules (`econml_utils.py`, `econml.API.py`, etc.)

---

## 3. Main notebooks to run

### Option A – Tutorial first (recommended)

1. Open **`econml.example.ipynb`**:

   * Run all cells from top to bottom.
   * This notebook:

     * Builds the merged NHANES dataset,
     * Defines treatment and outcomes,
     * Runs EconML DRLearner for SBP and fasting glucose,
     * Explores heterogeneity by BMI,
     * Compares EconML ATE vs OLS (bonus part of the assignment).

2. Optionally open **`econml.API.ipynb`**:

   * Shows the API-level functions in isolation.
   * Handy as a quick reference if you want to call the API from your own code.

### Option B – Script version

If you prefer a script instead of a notebook:

```bash
bash docker_bash.sh         # open a shell inside the container
python econml.example.py    # run the example pipeline as a script
```

The script mirrors the same steps as `econml.example.ipynb` but without the
narrative markdown.

---

## 4. Dependencies (inside Docker)

The Docker image already has Python and the required packages installed.
If you need to reinstall manually inside the container:

```bash
pip install -r requirements.txt
```

But in normal use for MSML610, simply building the Docker image and launching
Jupyter via the provided scripts should be enough.

---

That’s all that is needed to run and reproduce the results for this project.


