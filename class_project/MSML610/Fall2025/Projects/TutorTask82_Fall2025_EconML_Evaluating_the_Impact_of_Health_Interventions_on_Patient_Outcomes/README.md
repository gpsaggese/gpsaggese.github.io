
EconML – Evaluating the Impact of Health Interventions on Patient Outcomes (MSML610 – Fall 2025)
===============================================================================================

**Branch / Folder:** `TutorTask82_Fall2025_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes`

Project layout
--------------

```text
UmdTask82_EconML_Evaluating_the_Impact_of_Health_Interventions_on_Patient_Outcomes/
├── econml.API.ipynb          # Notebook describing the native API / design of the project
├── econml.API.md             # Markdown description of the API layer
├── econml.API.py             # Python API module (functions / classes to be called)
├── econml.example.ipynb      # End-to-end example notebook using the utils + API
├── econml.example.md         # Markdown writeup of the example pipeline
├── econml.example.py         # Python script version of the example
├── econml_utils.py           # Utility functions for data prep, models, evaluation
├── MSML610_DataPrepaparation_Karthik.ipynb   # Original data preparation notebook (reference)
├── Data_Preparation_Sri.ipynb               # Additional data preparation notebook (reference)
├── how_to_run.md             # Extra notes / instructions (optional)
├── requirements.txt          # Python dependencies for this project
├── Dockerfile                # Docker image definition
├── docker_build.sh           # Build the project Docker image
├── docker_bash.sh            # Open a shell inside the Docker container
├── docker_jupyter.sh         # Launch Jupyter Lab inside the container
├── changelog.txt             # Project version history
├── __init__.py
├── data/
│   └── raw/
│       └── .gitkeep          # Placeholder for raw datasets (not tracked by git)
├── tmp.build/                # Auto-generated helper directory (do not modify)
└── tutorial_github_data605_style/   # Original template files (do not modify)
````

## Quick start (Linux / macOS)

1. **Build Docker image**

```bash
bash docker_build.sh
```

2. **Open a shell in the container**

```bash
bash docker_bash.sh
# inside container:
python -V
pip list | grep econml
```

3. **Launch Jupyter Lab**

```bash
bash docker_jupyter.sh
# open the printed http://127.0.0.1:<PORT>/ URL in your browser
```

4. **Run the example notebook**

Open `econml.example.ipynb` in Jupyter and run it top-to-bottom.

It will eventually:

* Import functions from `econml_utils.py`
* Load and preprocess the health dataset
* Fit causal models using EconML
* Estimate the impact of specific interventions on patient outcomes
* Report summary metrics and visualizations

## What’s included (skeleton code)

* `econml_utils.py` – tiny library with placeholders for:

  * `load_health_data()` – load the health interventions dataset from `data/raw/`
  * `preprocess_data()` – clean features, handle missing values, encode categories
  * `split_train_test()` – create train / validation / test splits
  * `train_baseline_models()` – baseline predictive models (e.g. logistic regression, random forest)
  * `train_causal_model()` – EconML-based causal model (e.g. DRLearner, CausalForest)
  * `estimate_treatment_effects()` – compute ATE/CATE or uplift estimates
  * `evaluate_model()` – evaluation metrics and plots

* `econml.API.ipynb` – documents the API / design (inputs, outputs, assumptions).

* `econml.API.md` – text description of the same API and workflow.

* `econml.API.py` – Python API surface for calling the project from other code.

* `econml.example.ipynb` – main runnable example, showing the full pipeline:

  1. Load & preprocess data
  2. Train causal model(s)
  3. Estimate treatment effects
  4. Interpret and visualize results

* `econml.example.py` – script version of the example pipeline.

* Docker scripts:

  * `docker_build.sh`
  * `docker_bash.sh`
  * `docker_jupyter.sh`
