# Employee Attrition Prediction with LIME: Complete Application

## Overview

This document describes a complete end-to-end application that uses the
`lime_attrition_utils` API layer to build and explain an employee attrition
prediction model.

The example corresponds to the notebook `lime_attrition.example.ipynb` and shows
how to go from raw HR data to:

- A trained gradient-boosting–based classifier that predicts which employees are
  at risk of leaving.
- Local LIME explanations for individual employees.
- Aggregated LIME explanations that highlight global drivers of attrition.
- A “bonus” analysis that tests how explanations change when we restrict the
  model to different feature subsets (demographics, workload, compensation, etc.).

All modelling, preprocessing, and explanation logic is orchestrated via the
wrapper API defined in `lime_attrition_utils.py` and documented in
`lime_attrition.API.md`. The notebook itself only wires together the API calls
and performs light visualization.

---

## Application Goal

The goal of this application is to:

1. **Predict Employee Attrition**  
   Train a supervised learning model that estimates the probability that an
   employee will leave the company.

2. **Explain Predictions with LIME**  
   Use Local Interpretable Model-agnostic Explanations (LIME) to understand
   which factors drive the model’s prediction for a specific employee.

3. **Discover Global Drivers of Attrition**  
   Aggregate LIME explanations over multiple high-risk employees to identify
   common patterns and risk factors.

4. **Assess Explanation Stability**  
   Perform a **feature subset experiment** to see how explanations change when
   the model is restricted to different groups of features (demographics,
   workload, compensation & tenure, etc.).

5. **Provide Actionable Insights for HR**  
   Translate model and LIME outputs into high-level recommendations that
   HR teams could act on (e.g., addressing overtime, work–life balance,
   or tenure-related risk).

---

## Dataset

### Source: IBM HR Employee Attrition Dataset

The application uses the well-known IBM HR Employee Attrition dataset (often
distributed via Kaggle as “IBM HR Analytics Employee Attrition & Performance”).

- **Domain**: Human resources / workforce analytics  
- **Unit of analysis**: One row per employee  
- **Target label**: `Attrition` (Yes/No), converted to a binary target (1 = leave)  
- **Features** (examples):
  - **Demographics**: `Age`, `Gender`, `Education`, `MaritalStatus`, `EducationField`
  - **Job and role**: `JobRole`, `JobLevel`, `Department`, `BusinessTravel`
  - **Compensation**: `MonthlyIncome`, `PercentSalaryHike`, `StockOptionLevel`
  - **Tenure and experience**: `TotalWorkingYears`, `YearsAtCompany`,
    `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`
  - **Workload and satisfaction**: `OverTime`, `JobInvolvement`,
    `WorkLifeBalance`, `EnvironmentSatisfaction`, `JobSatisfaction`,
    `RelationshipSatisfaction`

### Data Characteristics

- **Size**: ~1,470 employees (in the standard version of the dataset)
- **Target imbalance**: Attrition is relatively rare (~15–20% churn), which
  motivates the use of metrics such as **precision–recall AUC** in addition to
  accuracy and ROC AUC.
- **Feature types**:
  - **Numeric** features (e.g., income, age, tenure).
  - **Categorical** features (e.g., job role, overtime flag, department).
- **Noise and missingness**: The utilities handle any minor issues, including
  type conversion and safe handling of missing values.

---

## API Layer: `lime_attrition_utils.py`

The example is designed to showcase the use of the project’s internal API layer
rather than writing ad-hoc modelling code in the notebook.

Key abstractions from `lime_attrition_utils.py` include:

- **Configuration Objects**
  - `AttritionDataConfig`: Centralizes dataset-related configuration such as
    the target column (`Attrition`), positive class label, and train/test split
    parameters.
  - `ModelConfig`: Controls which model families to train
    (GradientBoosting, XGBoost, LightGBM, RandomForest) as well as common
    hyperparameters (number of estimators, learning rate, max depth, etc.).
  - `LimeConfig`: Configures LIME explanations (number of features,
    number of samples, random seed).

- **Data Loading and Preparation**
  - `load_raw_attrition_data(path)`: Loads the raw CSV file.
  - `clean_attrition_data(df, cfg)`: Cleans the dataset and converts the
    attrition label into a binary target.
  - `split_features_target(df, cfg)`: Splits the cleaned DataFrame into
    `X` (features) and `y` (binary target).
  - `train_test_split_attrition(X, y, cfg)`: Performs a stratified train/test
    split with consistent random seeding.

- **Preprocessing and Modelling**
  - `build_preprocessor(X_train)`: Builds a scikit-learn `ColumnTransformer`
    that:
    - Standard-scales numeric features.
    - One-hot encodes categorical features.
  - `train_attrition_models(X_train, y_train, preprocessor, model_cfg)`: Trains
    one or more pipeline models (`preprocess` + `model`) according to
    `ModelConfig`.
  - `evaluate_models(models, X_test, y_test)`: Evaluates each trained model on
    accuracy, precision, recall, F1, ROC AUC, and PR AUC, and returns a metrics
    dictionary keyed by model name.

- **LIME Explanation Utilities**
  - `build_lime_explainer(preprocessor, X_train, class_names)`: Builds a LIME
    explainer on top of the **fitted** preprocessing pipeline and the
    preprocessed training data.
  - `explain_single_employee(explainer, model_pipeline, raw_row, preprocessor, lime_config)`:
    Generates a LIME explanation object for a single employee.
  - `batch_lime_explanations(...)`: Produces a compact HR-facing summary table
    of top LIME features for the highest-risk employees.
  - `batch_lime_explanations_long(...)`: Produces a “long” table where each row
    is a (employee, feature, weight) triple.
  - `aggregate_lime_features(long_df)`: Aggregates feature contributions across
    employees, computing summary statistics such as mean absolute weight.
  - `plot_lime_aggregate_bar(agg_df, ...)`: Convenience wrapper for plotting
    the most influential features by mean absolute LIME weight.

The `.example.ipynb` uses these utilities as building blocks; complex logic
remains in the module, keeping the notebook focused on the workflow and results.

---

## Complete Pipeline

### Step 1: Data Loading and Cleaning

**Objective**: Load the IBM HR attrition dataset and produce a clean, analysis-ready
DataFrame.

**Process** (as implemented in the example notebook):

1. Locate the dataset in a `data/` directory (e.g., `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`).
2. Instantiate `AttritionDataConfig`.
3. Call:
   - `df_raw = load_raw_attrition_data(path)`
   - `df = clean_attrition_data(df_raw, cfg)`
   - `X, y = split_features_target(df, cfg)`
4. Inspect the first few rows and compute the overall attrition rate.

**Rationale**:
- Centralizing file access and cleaning in the API layer ensures that the
  example notebook remains lightweight and that other notebooks can reuse the
  same logic consistently.

---

### Step 2: Exploratory Data Analysis (EDA)

**Objective**: Understand key patterns in attrition before building a model.

**Process**:

1. **Attrition Rate**  
   Compute and plot the share of employees who left vs. those who stayed.

2. **Categorical Drivers**  
   Use `categorical_attrition_table(df, column, cfg)` and simple bar plots to
   examine attrition rates by:
   - `OverTime`
   - `JobRole`
   - `Department`
   - `BusinessTravel`
   - `MaritalStatus`

3. **Numeric Distributions**  
   Plot histograms of numeric features split by attrition status, such as:
   - `MonthlyIncome`
   - `Age`
   - `DistanceFromHome`
   - `YearsAtCompany`
   - `TotalWorkingYears`

4. **Correlation Analysis and Heatmap (Numeric Features)**  
   - Create `num_df` with all numeric features plus `AttritionTarget`.
   - Compute correlations with the binary attrition target and display the most
     correlated features.
   - Plot a correlation heatmap over numeric features and `AttritionTarget` to
     visualize positive and negative relationships.

**Insights (qualitative)**:
- Overtime workers and certain job roles tend to show higher attrition.
- Longer distances from home and some tenure-related variables may correlate
  with increased risk.
- Numeric correlation plus group-wise categorical EDA provide a good initial
  view of potential drivers.

---

### Step 3: Feature Engineering

**Objective**: Create engineered features that may better capture relationships
between compensation, tenure, and risk.

**Engineered Features** (implemented in the notebook via `add_engineered_features`):

- `IncomePerYearAtCompany`  
  `MonthlyIncome / (YearsAtCompany + 1)`

- `TenureRatio`  
  `YearsAtCompany / (TotalWorkingYears + 1)`

- `LongCommute`  
  Binary indicator for employees whose `DistanceFromHome` exceeds a threshold
  (e.g., 10 units).

- `EarlyCareer`  
  Binary indicator for younger employees (e.g., `Age <= 30`).

These features are added on top of the original `X` to produce `X_fe`, which is
then fed into the preprocessing and modelling pipeline.

**Reason**:
- Ratios and binary indicators can capture non-linear and interaction-like
  effects without building a much more complex model.
- They are still easily interpretable in LIME explanations (e.g., “LongCommute”
  as a single feature).

---

### Step 4: Model Training and Evaluation

**Objective**: Train several model families and pick the one that performs best
on held-out test data.

**Process**:

1. **Train/Test Split**
   - Use `train_test_split_attrition(X_fe, y, cfg)` to create stratified
     `X_train`, `X_test`, `y_train`, `y_test`.

2. **Preprocessing**
   - Build a preprocessing pipeline via `build_preprocessor(X_train)` that:
     - Scales numeric features.
     - One-hot encodes categorical features.

3. **Model Training**
   - Configure `ModelConfig` to enable:
     - Scikit-learn GradientBoostingClassifier
     - XGBoost (if installed)
     - LightGBM and RandomForest (optional extras)
   - Call `train_attrition_models(X_train, y_train, preprocessor, model_cfg)` to
     obtain a dictionary of fitted pipelines.

4. **Evaluation**
   - Call `evaluate_models(models, X_test, y_test)` to compute:
     - Accuracy, precision, recall, F1
     - ROC AUC
     - Precision–Recall AUC (PR AUC)
   - Construct a table sorted by PR AUC and ROC AUC to select the **best model**.
   - The example notebook uses **PR AUC** as the primary selection metric,
     because the dataset is imbalanced and high precision at reasonable recall
     is important.

5. **Evaluation Curves**
   - For the best model (`best_pipe`), plot:
     - ROC curve (TPR vs FPR, with AUC).
     - Precision–Recall curve (precision vs recall, with average precision).

**Typical Outcome (qualitative)**:
- Gradient-boosted tree models (scikit-learn, XGBoost, LightGBM) generally
  achieve solid ROC AUC and PR AUC on this dataset.
- XGBoost often emerges as the top model by PR AUC, and is used for subsequent
  LIME analysis in the example.

---

### Step 5: LIME Explanations

**Objective**: Explain individual predictions and derive aggregated drivers of
attrition using LIME.

#### Step 5.1: Single High-Risk Employee

1. Compute predicted probabilities `probs = best_pipe.predict_proba(X_test)[:, 1]`.
2. Identify the index of the **highest-risk** test employee (`top_idx`).
3. Extract the corresponding raw feature row `row = X_test.iloc[top_idx]`.

4. Build a LIME explainer:

   ```python
   pre_fitted = best_pipe.named_steps["preprocess"]
   explainer = build_lime_explainer(
       preprocessor=pre_fitted,
       X_train=X_train,
       class_names=["Stay", "Leave"],
   )
   ```

5. Configure LIME via `LimeConfig(num_features=10, num_samples=5000)`.

6. Call `explain_single_employee(...)` to obtain a LIME explanation for this
   employee and visualize:
   - A ranked list of feature contributions (feature, weight).
   - A bar plot from `exp.as_pyplot_figure()`.

This shows which features in this employee’s profile push the model toward
“Leave” versus “Stay”.

#### Step 5.2: LEAVE vs STAY Comparison

To illustrate both ends of the decision threshold, the notebook:

1. Finds:
   - One test example predicted to **leave**.
   - One test example predicted to **stay**.
2. Computes LIME explanations for both and prints the feature–weight lists.
3. Plots side-by-side LIME bar plots, highlighting how:
   - Risky features (e.g., high overtime, long commute, certain job roles)
     drive the “Leave” prediction.
   - Protective features (e.g., higher satisfaction, lower distance, more
     moderate workload) may support the “Stay” prediction.

#### Step 5.3: Batch Explanations and Aggregated Drivers

To move from individual to global insights:

1. Use `batch_lime_explanations(...)` on the top N (e.g., 25) highest-risk
   employees to generate a compact summary table with:
   - Employee ID or index.
   - Predicted probability of attrition.
   - Top K features and their LIME weights.

2. Use `batch_lime_explanations_long(...)` to create a “long” table of
   (employee, feature, weight).

3. Call `aggregate_lime_features(long_df)` to produce an aggregate driver table
   with summary statistics such as:
   - mean LIME weight,
   - mean absolute LIME weight,
   - frequency with which a feature appears as important.

4. Optionally call `plot_lime_aggregate_bar(agg_df, top_n=12, sort_by="mean_abs_weight")`
   to visualize the **most influential features** across high-risk employees.

**Insights (qualitative)**:
- Features such as `OverTime`, `JobRole`, `MonthlyIncome`, `YearsAtCompany`,
  `DistanceFromHome`, and satisfaction scores frequently appear as strong
  contributors to attrition risk.
- Engineered features like `LongCommute` or `TenureRatio` often appear with
  intuitive signs (e.g., long commute increasing risk).

---

### Step 6: Bonus – Feature Subset Experiments

**Objective**: Study how explanations change when the model is built from
different groups of features, and quantify this change using Jaccard similarity
on top LIME features.

**Process**:

1. **Reference Explanation**  
   - Use the full feature set (`All+FE`) to train the best model.
   - Record the top-10 LIME features (after light normalization of feature names)
     for the highest-risk employee as the **reference key set**.

2. **Define Feature Subsets**

   Conceptual subsets are defined as lists of columns:

   - **All+FE (reference)**  
     All original and engineered features.

   - **Demographics only**  
     Age, gender, education level and field, marital status, satisfaction
     scores, etc.

   - **Workload & work-life**  
     Overtime, business travel, job involvement, work–life balance,
     job role/level, department, distance from home, years in current role,
     years with current manager, and engineered flags (e.g., LongCommute).

   - **Compensation & tenure**  
     Monthly income, salary hike, stock options, total working years,
     years at company, years since last promotion, and engineered ratio
     features (IncomePerYearAtCompany, TenureRatio).

3. **Train a Model per Subset**

   For each subset:

   - Restrict `X_train` and `X_test` to the subset’s columns.
   - Build a new preprocessor and model configuration.
   - Train models via `train_attrition_models`.
   - Evaluate via `evaluate_models` and select the best subset model
     (preferably using the same family as the global best model when possible).

4. **Explain the Same Employee Under Each Subset Model**

   - For each subset, explain the **same** high-risk employee index using the
     subset model and its LIME explainer.
   - Extract the top-10 LIME features and map them to stable feature keys.

5. **Compute Jaccard Similarity**

   - For each subset, compute Jaccard similarity between:
     - Top-10 feature keys from the subset model, and
     - Top-10 feature keys from the full reference model.
   - Store:
     - Subset name
     - Trained model type
     - ROC AUC / PR AUC
     - Predicted probability of attrition for the same employee
     - Jaccard similarity score

6. **Visualization and Interpretation**

   - Plot a bar chart of Jaccard similarity vs. subset.
   - Inspect the table `subset_results` sorted by Jaccard score.

**Interpretation (qualitative)**:
- High Jaccard similarity means explanations are fairly **stable** when
  restricting to that subset.
- Lower similarity indicates that important features in the full model rely on
  information not present in that subset (e.g., workload-only models may miss
  some demographic or tenure interactions).
- Comparing performance metrics (ROC AUC, PR AUC) side-by-side with Jaccard
  similarity helps assess the trade-off between model simplicity, performance,
  and explanation stability.

---

## Design Decisions

### Why Gradient Boosting / XGBoost?

- **Non-linear interactions**: Tree-based methods naturally handle feature
  interactions and non-linear effects common in HR data (e.g., the interaction
  of tenure and role level).
- **Robustness**: They perform well out-of-the-box on mixed-type tabular data.
- **Explainability**: While global interpretation can be complex, local
  explanations via LIME are straightforward and intuitive.

### Why LIME?

- **Model-agnostic**: LIME works with any classifier, including ensembles like
  XGBoost.
- **Local explanations**: It answers: “Why did the model make *this* prediction
  for *this* employee?”
- **HR alignment**: Produces human-readable rules (e.g., “OverTime=Yes”,
  “MonthlyIncome ≤ X”), which align well with HR practitioners’ mental models.

### Why Engineered Features?

- **Ratios and flags** (e.g., `IncomePerYearAtCompany`, `TenureRatio`,
  `LongCommute`):
  - Capture concepts (commute difficulty, tenure vs experience, adjusted income)
    that may not be apparent from raw feature values alone.
  - Remain simple enough to appear naturally and meaningfully in LIME
    explanations.

### Why PR AUC as Primary Model Selection Metric?

- The dataset is **imbalanced**; predicting everyone as “stay” can give a high
  accuracy but useless predictions.
- **PR AUC** focuses on the model’s ability to identify attrition cases
  (positive class) with good precision at useful levels of recall.
- ROC AUC is still reported for completeness, but PR AUC drives the choice of
  the “best” model for downstream explanation.

---

## Results Interpretation

### Understanding the Model Outputs

- **Prediction scores**: For each employee, the model outputs a probability of
  leaving; thresholds can be tuned depending on HR’s tolerance for false
  positives vs false negatives.
- **Evaluation curves**: ROC and Precision–Recall curves illustrate how
  performance changes across different thresholds.

### Understanding LIME Explanations

- **Single-employee explanations** show the most influential features for that
  employee, with positive weights pushing toward “Leave” and negative weights
  toward “Stay”.
- **LEAVE vs STAY examples** highlight how risk factors differ between a
  high-risk and a low-risk profile.
- **Aggregate LIME drivers** summarize which features consistently appear as
  high-impact across high-risk employees.

### Bonus: Stability of Explanations Across Feature Subsets

- The **Jaccard similarity** analysis indicates whether the most important
  features for a given prediction remain the same when restricting the model to
  specific conceptual subsets (demographics, workload, compensation/tenure).
- HR teams can use this to reason about:
  - Which information is essential for reliable explanations.
  - Whether simpler models (using fewer feature groups) might be acceptable
    trade-offs between transparency and performance.

---

## Limitations and Future Improvements

### Current Limitations

1. **Static Snapshot**: The dataset is cross-sectional; it does not account for
   time-varying factors or events (e.g., recent organizational changes).
2. **Correlational, Not Causal**: LIME explanations are descriptive of the
   model’s behavior, not causal effects of interventions.
3. **Single Dataset**: Insights are specific to the IBM HR dataset and may not
   generalize directly to another organization without retraining.
4. **Manual Feature Subsets**: The feature subsets used in the bonus experiment
   are manually defined based on domain intuition.

### Future Improvements

1. **Temporal Modelling**: Incorporate time-based features or longitudinal data.
2. **Causal Analysis / Uplift Modelling**: Explore methods that estimate the
   impact of interventions (e.g., salary adjustment, remote work).
3. **Automated Feature Grouping**: Use clustering or domain ontologies to
   define feature subsets more systematically.
4. **Interactive Dashboards**: Integrate LIME and model outputs into an
   interactive HR dashboard for non-technical stakeholders.
5. **Fairness and Bias Audits**: Extend the pipeline to check for and mitigate
   unfair biases across demographic groups.

---

## Conclusion

This example demonstrates a complete end-to-end pipeline for Employee Attrition
Prediction using the `lime_attrition_utils` API layer:

- **From raw data to clean dataset** using standardized loading and cleaning
  utilities.
- **From features to model** through a reusable preprocessing and training
  pipeline that supports multiple gradient-boosting–based models.
- **From scores to explanations** using LIME to illuminate individual and
  aggregated drivers of attrition risk.
- **From explanations to robustness checks** via feature subset experiments that
  probe the stability of explanations under different information sets.

The combination of a clean API layer, a focused example notebook
(`lime_attrition.example.ipynb`), and this narrative document makes the project
reproducible, understandable, and ready for adaptation to other HR attrition
contexts.
