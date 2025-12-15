# XGBoost and SHAP API Demonstration: A Walkthrough of `shap_credit_API.ipynb`

## Overview

The `shap_credit_API.ipynb` notebook is a foundational companion to the credit scoring project. This notebook focuses on demonstrating the core APIs—**XGBoost** for gradient boosting and **SHAP** for explainability—before applying them to the full pipeline. It serves as a technical deep-dive into the tools and the `credit_scoring_shap` package structure.

Designed with beginners and code-explorers in mind, this notebook breaks down how XGBoost and SHAP work on simple toy data first, then introduces the German Credit dataset and shows how the project's modular API wraps these tools for production use. It's the perfect starting point for understanding the building blocks before seeing the complete system in `shap_example.ipynb`.

## Purpose

The primary purpose of `shap_credit_API.ipynb` is to introduce and demonstrate the core API functions used in the credit scoring pipeline. Its key objectives are to:

- Showcase XGBoost's capabilities on synthetic toy data to build intuition about gradient boosting.
- Demonstrate native SHAP TreeExplainer usage and visualizations (summary plots) on simple data.
- Introduce the German Credit dataset and show the preprocessing pipeline that transforms raw features into model-ready inputs.
- Explain the `credit_scoring_shap` package structure: how data, modeling, evaluation, and explanation modules separate concerns cleanly.
- Provide docstring references and usage patterns so you can extend or customize the pipeline.
- Connect the dots between individual APIs (XGBoost, SHAP) and the integrated pipeline (how they work together).

This notebook is an educational resource focused on API understanding, preparing users for the comprehensive analysis in `shap_example.ipynb`.

## Quick reference: the native XGBoost and SHAP APIs used here

### XGBoost core calls
- `xgb.XGBClassifier(**params)`
  Creates a gradient boosted tree model for classification.
- `model.fit(X_train, y_train)`
  Trains the model.
- `model.predict_proba(X)[:, 1]`
  Returns probabilities for class `y=1`.
- `model.feature_importances_`
  Returns a global importance score per feature, but it does not show direction.

### SHAP core calls
- `shap.TreeExplainer(model)`
  Creates a SHAP explainer for a trained tree model.
- `explainer.shap_values(X)`
  Computes per feature contributions for each row.
- `shap.summary_plot(shap_values, X, feature_names=...)`
  Creates a global view of which features matter most and how they push predictions.


## Notebook Structure

The notebook is organized into six conceptual sections, building from motivation to full API documentation:

### Section 1: Why Credit Scoring Needs Explainability

- **Purpose**: Set context before diving into code—understand *why* explainability matters in finance.
- **Process**: Discusses stakeholder needs: regulators require fair, transparent decisions; customers want to know why they were denied; model owners need to validate that models are sensible.
- **Output**: Conceptual framing—XGBoost provides accuracy (strong tabular performance), SHAP provides explanations (feature attributions), and together they meet the dual requirements of performance and interpretability.
- **Insights**: In regulated industries like banking, explainability isn't optional. Laws like ECOA (Equal Credit Opportunity Act) require lenders to explain adverse decisions. A model that's 99% accurate but unexplainable may be legally unusable.

### Section 2: Quick XGBoost Model on Toy Data (Cells 1-4)

#### Imports and Setup

- **Purpose**: Load all necessary libraries and configure paths for the project.
- **Process**: Imports pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, and project-specific modules from `credit_scoring_shap`. Sets up plot defaults and creates a `TrainingConfig` object.
- **Output**: Config object displayed showing `DataConfig`, `ModelConfig`, and paths.
- **Insights**: The `TrainingConfig` centralizes all hyperparameters and paths—changing learning rate or max_depth happens in one place, not scattered across code. This is production best practice.

#### Create Synthetic Dataset

- **Purpose**: Generate a simple 10-feature, 500-sample binary classification dataset to demonstrate XGBoost's basic API.
- **Process**: Uses `make_classification` with controlled parameters (5 informative features, 2 redundant, 2 clusters per class) and splits 80-20 into train/test with stratification.
- **Output**: Training shape `(400, 10)`, test shape `(100, 10)`.
- **Insights**: Toy data is pedagogically powerful—it runs in seconds, patterns are interpretable, and you can validate that APIs work correctly before tackling messy real data. The 10 features are just named `x0` through `x9` so there's no domain confusion.

#### Train Native XGBoost Classifier

- **Purpose**: Show XGBoost's native API without any project wrappers.
- **Process**: Instantiates `xgb.XGBClassifier` with explicit hyperparameters (200 trees, learning rate 0.1, max depth 4, subsampling 0.9), fits on toy training data, predicts probabilities, and computes AUC.
- **Output**: AUC of approximately 0.97 on toy data. Very high because the synthetic data is cleanly separable.
- **Insights**: The high AUC (0.97) is a sanity check that XGBoost works. On real credit data, expect 0.75-0.80. Each hyperparameter serves a purpose: `n_estimators` controls ensemble size, `learning_rate` controls step size (shrinkage), `max_depth` limits tree complexity, `subsample`/`colsample_bytree` add randomness for regularization.

#### Feature Importances

- **Purpose**: Show XGBoost's built-in feature importance scores as a baseline before introducing SHAP.
- **Process**: Extracts `model.feature_importances_`, sorts descending, and prints the top 5 features.
- **Output**: Something like `x0: 0.267, x5: 0.153, x1: 0.138, ...`
- **Insights**: Feature importances tell you *which* features matter but not *how*—does high x0 increase or decrease predictions? Are effects consistent across samples? This limitation is why we need SHAP.

### Section 3: Native SHAP Demonstration

#### SHAP Summary Plot on Toy Model

- **Purpose**: Introduce SHAP by creating a summary plot that shows both importance and directionality.
- **Process**: Creates a `shap.TreeExplainer` for the toy XGBoost model, computes SHAP values on a 200-sample background dataset, and generates a beeswarm summary plot. Handles the binary classification quirk where `shap_values` returns a list (picks positive class).
- **Output**: A SHAP summary plot saved to `reports/api_toy_shap_summary.png` and displayed inline. Shows features on y-axis (ranked by importance), SHAP values on x-axis (negative pushes toward class 0, positive toward class 1), with dots colored by feature value (blue=low, red=high).
- **Insights**: The summary plot is SHAP's signature visualization. Each dot is a sample—wide spread means variable impact. Color reveals directionality: if red dots (high feature value) cluster on the right (positive SHAP), then high values increase predictions. This single plot combines feature importance, effect direction, and distribution—far richer than XGBoost's scalar importances.

### Section 4: German Credit Dataset (Cells 6-7)

#### Load Raw German Credit Data

- **Purpose**: Switch from toy data to the real dataset used in the main pipeline.
- **Process**: Calls `load_raw_data(cfg.data)` which fetches the German Credit dataset from UCI ML Repository and returns a pandas DataFrame.
- **Output**: Raw shape `(1000, 21)`. First 5 rows displayed showing columns like `status_checking_account`, `duration_months`, `credit_amount`, `purpose`, etc.
- **Insights**: The 1000 rows represent loan applications. The 21 features include categorical (checking account status, credit history, employment) and numerical (credit amount, duration, age). The target is binary: 1=Good (loan repaid), 0=Bad (defaulted). Scanning the raw data builds intuition about what the model will see after preprocessing.

#### Load Preprocessed Data

- **Purpose**: Show the transformation from raw to model-ready data.
- **Process**: Calls `load_and_preprocess(cfg.data)` which one-hot encodes categorical features, standardizes numerical features, and splits 80-20 into train/test with stratification.
-- **Output**: Encoded training shape `(800, 61)`, test shape `(200, 61)`. Train positive rate: ~30% (Bad), test positive rate: ~30% (Bad). First 10 feature names displayed (mix of numerical like `duration_months` and one-hot encoded like `status_checking_account_A11`).
- **Insights**: The 21→61 feature explosion is normal with one-hot encoding. For example, `status_checking_account` with 4 categories becomes 4 binary indicators. The 70-30 class imbalance (70% good loans) will challenge models later. The preprocessing pipeline encapsulates all transformations in a reusable object.

### Section 5: Project Modeling API (Cells 8-10)
### The `credit_scoring_shap` Package

The project is organized as a Python package with clear separation of concerns:

**Core Modules:**
- **`config.py`**: Defines `DataConfig`, `ModelConfig`, and `TrainingConfig` dataclasses that centralize all configuration parameters (test size, random seeds, hyperparameters, report paths)

- **`data.py`**: Handles data loading and preprocessing
  - `load_raw_data()`: Fetches German Credit dataset from UCI via `ucimlrepo`
  - `load_and_preprocess()`: Performs one-hot encoding, standardization, and stratified train/test split
  - Returns preprocessed DataFrames and fitted preprocessing pipeline

- **`modeling.py`**: Model building and evaluation
  - `build_model()`: Constructs XGBoost classifier from `ModelConfig`
  - `train_model()`: Fits the model on training data
  - `evaluate_model()`: Computes AUC, confusion matrix, classification report

- **`evaluation.py`**: Visualization and metrics reporting
  - `plot_confusion_matrix()`: Saves confusion matrix heatmap
  - `plot_roc_curves()`: Generates ROC and Precision-Recall curves
  - `save_metrics_text()`: Persists metrics to text file

- **`explain.py`**: SHAP explainability
  - `build_shap_explainer()`: Creates TreeExplainer and computes SHAP values
  - `plot_global_shap_summary()`: Generates bar and beeswarm plots
  - `plot_shap_dependence_for_top_feature()`: Creates dependence plot
  - `plot_shap_decision_for_index()`: Produces decision plots for individual instances

- **`sensitivity.py`**: "What-if" analysis
  - `run_sensitivity_for_instance()`: Varies top features and plots probability changes

#### Build Model from Configuration

- **Purpose**: Demonstrate the project's `build_model()` function which constructs XGBoost from config rather than hardcoding parameters.
- **Process**: Calls `build_model(cfg.model)` which reads hyperparameters from the `ModelConfig` dataclass.
- **Output**: Displays the XGBClassifier object with all hyperparameters shown (learning_rate=0.05, max_depth=4, n_estimators=400, etc.).
- **Insights**: Configuration-driven design means hyperparameters live in one place (`cfg.model`), not scattered through code. Want to experiment with different learning rates? Change `cfg.model.learning_rate`, not grep through notebooks. This scales to teams and production.

#### Train and Evaluate with Project API

- **Purpose**: Show the complete train-evaluate cycle using project wrapper functions.
- **Process**: Calls `train_model(model, X_train, y_train)` to fit, then `evaluate_model(model, X_test, y_test, threshold=0.5)` to compute AUC, confusion matrix, and classification report.
- **Output**: AUC ~0.755, confusion matrix `[[27, 33], [23, 117]]`, classification report with precision/recall for both classes.
- **Insights**: These wrapper functions encapsulate best practices (proper train/test workflow, comprehensive metrics). The baseline AUC (0.755) establishes that credit scoring is moderately difficult—much harder than toy data (0.97) but solvable. The confusion matrix reveals the model's weakness: only 27/60 bad loans caught (45% recall).

#### Examine Function Docstrings

- **Purpose**: Provide reference documentation for key API functions.
- **Process**: Prints docstrings for `evaluate_model`, `plot_confusion_matrix`, `plot_roc_curves`.
- **Output**: Text documentation showing function signatures, parameter descriptions, and return values.
- **Insights**: Good APIs are self-documenting. Each function has a docstring explaining what it does, what inputs it expects, and what outputs it produces. This is your reference when extending the code. Want to know what `evaluate_model` returns? Check its docstring.

### Section 6: Explainability and Sensitivity APIs

#### Build SHAP Explainer with Project API

- **Purpose**: Show how the project wraps SHAP TreeExplainer for convenience.
- **Process**: Calls `build_shap_explainer(model, X_train, feature_names, cfg.reports_dir)` which creates the explainer, pre-computes SHAP values on training data, and handles binary classification quirks.
- **Output**: Returns `(explainer, shap_values)` tuple. The explainer can be reused for new samples; the shap_values are pre-computed for visualization.
- **Insights**: Pre-computing SHAP values on the full training set (800 samples) takes a few seconds but saves time when making multiple plots. The function handles the binary classification edge case where `shap_values` can be a list or array, simplifying downstream code.

#### Print API Docstrings for Explanation Functions

- **Purpose**: Document the explanation and sensitivity modules' APIs.
- **Process**: Prints docstrings for `build_shap_explainer`, `run_sensitivity_for_instance`.
- **Output**: Text showing these functions' signatures and purposes.
- **Insights**: The sensitivity function is powerful but niche—most users only need global SHAP summaries. For edge cases (explaining specific denials to customers), sensitivity analysis shows "what would need to change" for approval. The API makes this accessible without deep SHAP knowledge.

## Common pitfalls

- `predict_proba(X)[:, 1]` always refers to class `y=1`.
   In this project, `y=1` means Bad and `y=0` means Good.
- In binary classification, `shap_values` can be returned as a list.
  If that happens, use index 1 to get SHAP values for class `y=1`.
- XGBoost feature importances are useful for ranking features, but they do not tell you direction.
  Use SHAP when you need to explain why a specific prediction moved up or down.


## Educational Value

This notebook is valuable for several reasons:

- **Bottom-Up Learning**: Starts with toy examples where everything is transparent, then graduates to real complexity. This builds solid foundations.
- **Native vs. Wrapped APIs**: Shows both XGBoost's raw API and the project's wrappers, so you understand what the wrappers provide (convenience) and hide (boilerplate).
- **Code Architecture**: Explains the modular structure (data, modeling, evaluation, explanation as separate modules), teaching software engineering principles alongside ML.
- **Documentation Practices**: Emphasizes docstrings, configuration objects, and clear function signatures—production code skills often missing from tutorials.
- **Preparation for Main Pipeline**: After this notebook, the main `shap_example.ipynb` makes sense because you already understand the building blocks.

## How to Use

To run this notebook:

1. Install dependencies (same as main project):
   ```bash
   pip install pandas numpy scikit-learn xgboost shap matplotlib jupyter # add this in the scripts/install_common_packages.sh
   ```

2. Open the notebook:
   ```bash
   jupyter notebook shap_credit_API.ipynb
   ```

3. Run cells sequentially.

4. **Recommended Path**:
   - **Beginners**: Read Section 1 (motivation), run Sections 2-3 (toy examples), skim 4-6 on first pass, return after the main notebook.
   - **Experienced users**: Skim 1-3, focus on 4-6 (project-specific API details), read docstrings carefully.

### Requirements

- Python 3.8+
- Same dependencies as `shap_example.ipynb`

## Key Takeaways

- **XGBoost is flexible**: Hyperparameters control complexity (max_depth), speed (learning_rate), and regularization (subsample). Defaults are reasonable but not optimal.
- **SHAP improves on feature importances**: XGBoost's importances are scalar rankings; SHAP adds directionality (does high value help or hurt?) and per-sample granularity.
- **TreeExplainer is specialized**: Optimized for tree ensembles (XGBoost, Random Forest, LightGBM). For other model types (linear, neural nets), use different SHAP explainers.
- **Modular design enables flexibility**: Want to swap XGBoost for LightGBM? Change one function (`build_model`). Want to add LIME alongside SHAP? Add one module (`lime.py`). Separating concerns makes code maintainable.
- **Configuration centralizes decisions**: Hyperparameters, paths, and settings live in `TrainingConfig`, not scattered through code. This reduces bugs and makes experiments reproducible.
- **Toy examples accelerate learning**: The 10-feature synthetic data runs in seconds and patterns are clear. Master APIs on simple data before tackling complex real data.

## Connection to Main Notebook

Think of the two notebooks as:
- **This notebook (API)**: "Here are the tools in your toolkit—XGBoost, SHAP, and project modules"
- **Main notebook (Example)**: "Here's how to use those tools to build a complete credit scoring system"

You can learn the tools first (this notebook) then see the application (main notebook), or see the application first then come back to understand the tools. Both paths work.

**API as Foundation**: Everything in `shap_example.ipynb` rests on this API—`load_and_preprocess` provides clean data, `build_model` and `train_model` provide trained XGBoost, `evaluate_model` provides metrics, `build_shap_explainer` provides explanations. If the API is broken, the pipeline fails. That's why we validate it separately on simpler data.

For the complete credit scoring story with business context and real-world challenges, proceed to `shap_example.ipynb`.

## References

### Core libraries
1. XGBoost documentation
   https://xgboost.readthedocs.io/

2. SHAP documentation
   https://shap.readthedocs.io/

3. scikit-learn documentation
   https://scikit-learn.org/stable/

### scikit-learn APIs used in this notebook
4. make_classification
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

5. train_test_split
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

6. ROC AUC score
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

### SHAP API used in this notebook
7. TreeExplainer
   https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

8. SHAP summary plots
   https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/summary_plot.html

### Dataset
9. Statlog German Credit Data (UCI)
   https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

### Background papers
10. XGBoost paper: Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)
    https://arxiv.org/abs/1603.02754

11. SHAP paper: Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
    https://arxiv.org/abs/1705.07874
