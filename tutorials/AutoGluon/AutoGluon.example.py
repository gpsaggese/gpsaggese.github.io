# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AutoGluon: Predicting Employee Attrition
#
# - This notebook builds an attrition model on the IBM HR Analytics dataset, from
#   the raw CSV to a decision rule that HR could act on
# - The pedagogical arc:
#   - Load the data and set a logistic regression baseline
#   - Train an AutoGluon ensemble and compare it with the baseline, with
#     confidence intervals
#   - Move the decision threshold to trade precision for recall, using
#     out-of-fold predictions instead of the test set
#   - Explain the model globally (permutation importance) and per employee (SHAP)
#   - Summarize the predicted risk by department and job role
# - The task follows "Project 2: Employee Attrition Prediction" in
#   `class_project/project_descriptions/MSML610/AutoGluon_Project_Description.md`

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import autogluon.tabular as agt
import numpy as np
import pandas as pd

# %%
import helpers.hnotebook as hnotebook

import AutoGluon_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)
# Hide the GPU probe warnings (no `torch` in the image) and the SHAP internals.
utils.silence_noisy_loggers()

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

print("autogluon.tabular version: ", agt.__version__)

# %% [markdown]
# # Part 1: Load and Explore the Data

# %% [markdown]
# ## Cell 1.1: Download the Dataset
#
# - The file is the Kaggle "IBM HR Analytics Employee Attrition" dataset, read
#   from a pinned commit of IBM's GitHub repo, so no Kaggle account is needed
#   - The download is cached in `data/` and checked against a SHA-256 digest
# - IBM data scientists created the data: the employees are fictional
# - One row per employee, `Attrition` is "Yes" when the employee left

# %%
# Download the CSV (once) and load it.
csv_path = utils.download_attrition_data("data")
raw_df = pd.read_csv(csv_path)
print("raw_df.shape=", raw_df.shape)
display(raw_df.head())

# %% [markdown]
# ## Cell 1.2: Drop the Columns That Carry No Signal
#
# - AutoGluon drops constant columns by itself
#   - Dropping them here keeps the tables below short
# - `EmployeeNumber` is an identifier
#   - AutoGluon cannot know that and would treat it as a number
#   - Always remove IDs before training

# %%
# Find the columns with a single value.
n_unique = raw_df.nunique()
constant_cols = list(n_unique[n_unique == 1].index)
print("constant_cols=", constant_cols)

# %%
# Remove the constant columns and the identifier.
label = "Attrition"
positive_class = "Yes"
attrition_df = raw_df.drop(columns=constant_cols + ["EmployeeNumber"])
print("attrition_df.shape=", attrition_df.shape)

# %% [markdown]
# ## Cell 1.3: Look at the Label
#
# - About 1 employee in 6 left: the classes are imbalanced
#   - A model that always answers "No" has an accuracy of about 84%
#   - Accuracy is a poor metric here: ROC-AUC, recall, and precision tell more

# %%
# Count the employees who stayed and left.
display(attrition_df[label].value_counts().to_frame("n_employees"))

# %%
# Compare the attrition rate of employees with and without overtime.
utils.plot_rate_by_category(
    attrition_df, "OverTime", label, positive_class=positive_class
)
# Outcome: employees working overtime leave about 3 times as often.

# %% [markdown]
# # Part 2: Set a Baseline

# %% [markdown]
# ## Cell 2.1: Split Train and Test
#
# - 80% of the rows train the models, 20% are kept for the final evaluation
# - The split is stratified on the label, so both sets keep the 16% attrition rate
# - The test set has fewer than 300 employees, about 47 of whom left
#   - Every test metric below is noisy (Cell 3.3)

# %%
# Split the data, keeping the attrition rate equal in the two sets.
train_df, test_df = utils.split_train_test(attrition_df, stratify_col=label)
print("train_df.shape=", train_df.shape)
print("test_df.shape=", test_df.shape)
print("n_left (test)=", (test_df[label] == positive_class).sum())

# %% [markdown]
# ## Cell 2.2: Train a Logistic Regression
#
# - The baseline is what a data scientist writes by hand in a few minutes
#   - One-hot encoding of the string columns
#   - Scaling of the numerical columns
#   - A logistic regression on top
# - The code is in `utils.train_logistic_baseline()`

# %%
# Fit the baseline and predict the probability of leaving on the test set.
baseline = utils.train_logistic_baseline(train_df, label)
positive_idx = list(baseline.classes_).index(positive_class)
baseline_proba = baseline.predict_proba(test_df.drop(columns=[label]))[
    :, positive_idx
]
baseline_metrics = utils.compute_binary_metrics(
    test_df[label], baseline_proba, positive_class=positive_class
)
display(pd.Series(baseline_metrics, name="baseline").to_frame())
# Outcome: a ROC-AUC around 0.8, but at the 0.5 threshold the model flags only
# about 1 leaver in 3 (low recall).

# %% [markdown]
# # Part 3: Train AutoGluon

# %% [markdown]
# ## Cell 3.1: Fit an Ensemble
#
# - `eval_metric="roc_auc"`: the models are ranked on how well they order the
#   employees by risk
#   - The threshold is chosen later, in Part 4
# - `positive_class="Yes"`: probabilities and recall refer to leaving
# - `presets="good_quality"`
#   - Bags every model on 8 folds and fits a weighted ensemble on top
#   - Refits the chosen models on all the training rows for fast inference
# - `dynamic_stacking=False` skips a check that trains everything twice to
#   decide whether stacking overfits
#   - It doubles the training time and is not needed on 1,200 rows
# - The neural networks are excluded: they need `torch`, not installed in the
#   image

# %%
# Train the AutoGluon ensemble.
predictor = agt.TabularPredictor(
    label=label,
    eval_metric="roc_auc",
    positive_class=positive_class,
    path="tmp.autogluon/attrition",
    verbosity=1,
)
_ = predictor.fit(
    train_df,
    presets="good_quality",
    dynamic_stacking=False,
    excluded_model_types=["NN_TORCH", "FASTAI"],
    time_limit=300,
)
print("predictor.model_best=", predictor.model_best)

# %% [markdown]
# ## Cell 3.2: Read the Leaderboard
#
# - `_BAG_L1` models were trained on 8 folds and have a validation score
#   (`score_val`) computed out of fold
# - `_FULL` models are the same models refit on all the training rows
#   - They have no validation score of their own
#   - `predictor.model_best` is the refit ensemble
# - The ranking on `score_test` differs from the ranking on `score_val`
#   - With 47 leavers in the test set, a few rows decide the order
#   - Pick models on `score_val`, as AutoGluon does, never on `score_test`

# %%
# Score every model on the test set.
leaderboard_df = predictor.leaderboard(test_df)
cols = ["model", "score_test", "score_val", "stack_level", "fit_time"]
display(leaderboard_df[cols])

# %%
# Print the weight of each model in the ensemble.
ensemble_info = predictor.info()["model_info"]["WeightedEnsemble_L2"]
print("weights=", ensemble_info["children_info"]["S1F1"]["model_weights"])
# Outcome: the ensemble mixes 2 LightGBM variants and Extra Trees.

# %% [markdown]
# ## Cell 3.3: Compare With the Baseline, With Confidence Intervals
#
# - A single test ROC-AUC on 294 rows can move by several points with another
#   split
# - The bootstrap resamples the test set 1,000 times and reports the range that
#   contains 95% of the scores
# - If the two intervals overlap widely, the test set cannot tell the two models
#   apart

# %%
# Predict the probability of leaving with AutoGluon.
ag_proba = predictor.predict_proba(test_df)[positive_class].to_numpy()

# Compare the ROC-AUC of the two models with bootstrap intervals.
auc_df = pd.DataFrame(
    {
        "logistic_baseline": utils.bootstrap_roc_auc(
            test_df[label], baseline_proba, positive_class=positive_class
        ),
        "autogluon": utils.bootstrap_roc_auc(
            test_df[label], ag_proba, positive_class=positive_class
        ),
    }
).T.round(3)
display(auc_df)

# %% [markdown]
# - On this split the logistic baseline scores slightly higher, and the two
#   intervals overlap almost completely: the test set cannot separate them
#   - 1,200 rows with mostly monotonic effects (overtime, income, tenure) leave
#     little room for tree ensembles to beat a linear model
#   - AutoGluon reached the same level with no feature engineering, after
#     searching several model families
#   - Adding AutoGluon's own linear model (`"LR": {}` in `hyperparameters`) to
#     the ensemble does not change the test ROC-AUC on this split, which points
#     to a ceiling of the data rather than of the models
# - Keep a simple baseline in every AutoML project: it tells you whether the
#   complexity pays for itself

# %% [markdown]
# # Part 4: Choose the Decision Threshold

# %% [markdown]
# ## Cell 4.1: The 0.5 Threshold Misses Most Leavers
#
# - `predict()` flags an employee when the probability of leaving is at least 0.5
# - With a 16% base rate few employees pass 0.5, so recall is low
# - The spec asks for high recall: missing an employee who leaves is the costly
#   error

# %%
# Compute the test metrics of AutoGluon at the default threshold.
default_metrics = utils.compute_binary_metrics(
    test_df[label], ag_proba, positive_class=positive_class
)
display(pd.Series(default_metrics, name="autogluon_at_0.5").to_frame())

# %% [markdown]
# ## Cell 4.2: Calibrate the Threshold on Out-of-Fold Predictions
#
# - Choosing the threshold on the test set would leak it into the model
# - The bagged models give one out-of-fold probability per training row
#   - Each probability comes from models that did not see that row
#   - It is a fair stand-in for unseen data
# - `calibrate_decision_threshold()` searches the threshold that maximizes a
#   metric on these predictions
#   - Here F1, the balance of precision and recall

# %%
# Search the threshold that maximizes F1 on the out-of-fold predictions.
f1_threshold = predictor.calibrate_decision_threshold(metric="f1", verbose=False)
print("f1_threshold=", round(f1_threshold, 3))

# %% [markdown]
# ## Cell 4.3: Put a Cost on Each Error
#
# - F1 treats a missed leaver and a false alarm as equally bad
# - Assume that losing an employee costs 5 times a retention conversation
#   - `fn_cost=5`: an employee leaves without being flagged
#   - `fp_cost=1`: HR talks to an employee who would have stayed
#   - The ratio is an assumption: replace it with your own numbers
# - The cost-optimal threshold is the one with the lowest average cost on the
#   out-of-fold predictions

# %%
# Compute the metrics and the cost of every threshold on the out-of-fold data.
oof_proba = predictor.predict_proba_oof()[positive_class]
curve_df = utils.compute_threshold_curve(
    train_df[label],
    oof_proba.to_numpy(),
    positive_class=positive_class,
    fn_cost=5,
    fp_cost=1,
)
cost_threshold = float(curve_df.loc[curve_df["cost"].idxmin(), "threshold"])
print("cost_threshold=", round(cost_threshold, 3))

# %%
# Plot precision, recall, F1, and cost against the threshold.
thresholds = {
    "default": 0.5,
    "max F1": f1_threshold,
    "min cost": cost_threshold,
}
utils.plot_threshold_curve(curve_df, marked_thresholds=thresholds)
# Outcome: lowering the threshold raises recall and lowers precision; the cost
# is lowest well below 0.5.

# %% [markdown]
# ## Cell 4.4: Evaluate the Three Thresholds on the Test Set
#
# - The thresholds were chosen without the test set, so these numbers are a fair
#   estimate
# - ROC-AUC does not change: it does not depend on the threshold

# %%
# Compare test metrics and cost for the three thresholds.
rows = {}
for name, threshold in thresholds.items():
    metrics = utils.compute_binary_metrics(
        test_df[label],
        ag_proba,
        positive_class=positive_class,
        threshold=threshold,
    )
    metrics["cost"] = utils.compute_expected_cost(
        test_df[label],
        ag_proba,
        positive_class=positive_class,
        threshold=threshold,
        fn_cost=5,
        fp_cost=1,
    )
    rows[name] = metrics
threshold_df = pd.DataFrame(rows).T.round(3)
display(threshold_df)

# %%
# Show the confusion matrix of each threshold.
utils.plot_confusion_matrices(
    test_df[label],
    ag_proba,
    positive_class=positive_class,
    thresholds=thresholds,
)
# Outcome: the lower thresholds catch most leavers, at the price of more false
# alarms.

# %% [markdown]
# ## Cell 4.5: Store the Threshold in the Predictor
#
# - `set_decision_threshold()` makes `predict()` use the new threshold
#   - It is saved with the predictor, so a deployed copy behaves the same

# %%
# Make `predict()` use the cost-optimal threshold.
predictor.set_decision_threshold(cost_threshold)
n_flagged = (predictor.predict(test_df) == positive_class).sum()
print("predictor.decision_threshold=", round(predictor.decision_threshold, 3))
print("n_flagged (test)=", n_flagged)

# %% [markdown]
# # Part 5: Explain the Model

# %% [markdown]
# ## Cell 5.1: Global View With Permutation Importance
#
# - Shuffle one column, measure the drop of ROC-AUC on the test set
# - A `p_value` below 0.05 means the drop is unlikely to be noise
# - Importance measures what the model relies on, not what causes attrition

# %%
# Compute the permutation importance on the test set.
importance_df = predictor.feature_importance(test_df, num_shuffle_sets=3)
display(importance_df.head(10))

# %%
# Plot the most important features.
utils.plot_feature_importance(importance_df)
# Outcome: `OverTime` is by far the most important feature, as Cell 1.3
# suggested.

# %% [markdown]
# ## Cell 5.2: Explain One Employee With SHAP
#
# - SHAP splits a single prediction into one contribution per feature
#   - Baseline: the average predicted probability on a background sample
#   - Baseline plus the contributions equals the prediction of this employee
# - Kernel SHAP only calls `predict_proba()`, so it explains the whole ensemble
#   - It is slow: it evaluates `n_samples` perturbed rows per explained row
#   - A background of 30 training rows keeps it fast

# %%
# Background data: the reference rows of Kernel SHAP.
#
# - SHAP replaces some features of the explained row with background values and
#   measures the change in the prediction
# - 30 rows is a coarse but fast estimate of the "average employee"
# - More rows give more stable contributions and cost linearly more time
features_df = train_df.drop(columns=[label])
test_features_df = test_df.drop(columns=[label])
background_df = features_df.sample(30, random_state=0)
print("background_df.shape=", background_df.shape)

# %%
# Explain the test employee with the highest predicted risk.
top_idx = int(np.argmax(ag_proba))
top_row_df = test_features_df.iloc[[top_idx]]
top_shap_df = utils.explain_with_kernel_shap(
    predictor, background_df, top_row_df, positive_class=positive_class
)
expected_value = top_shap_df.attrs["expected_value"]
print("predicted probability=", round(ag_proba[top_idx], 3))
print("expected_value=", round(expected_value, 3))
print(
    "expected_value + sum(contributions)=",
    round(expected_value + top_shap_df.iloc[0].sum(), 3),
)

# %%
# Plot the largest contributions for this employee.
utils.plot_shap_contributions(
    top_shap_df.iloc[0],
    top_row_df.iloc[0],
    expected_value=expected_value,
    title="Why the model flags this employee",
)
# Outcome: red bars push the risk up and blue bars push it down. For this
# employee all the top contributions push it up, led by `OverTime = Yes` and a
# young age with little work experience.

# %% [markdown]
# ## Cell 5.3: From Local to Global SHAP
#
# - Averaging the absolute contributions over many employees gives a global
#   ranking
# - It should broadly agree with the permutation importance of Cell 5.1
#   - Both describe the same model with different methods

# %%
# Explain 50 test employees and rank the features by average contribution.
sample_df = test_features_df.sample(50, random_state=0)
sample_shap_df = utils.explain_with_kernel_shap(
    predictor, background_df, sample_df, positive_class=positive_class
)
utils.plot_mean_abs_shap(sample_shap_df)

# %% [markdown]
# # Part 6: Summarize the Risk by Group

# %% [markdown]
# ## Cell 6.1: Risk by Department and Job Role
#
# - The predicted risk of each test employee is averaged by group
# - `actual_rate` is the observed attrition in the same test rows
#   - Groups with few employees have noisy rates: check `n_employees`

# %%
# Average the predicted risk by department.
display(
    utils.summarize_risk_by(
        test_df, ag_proba, "Department", label, positive_class=positive_class
    )
)

# %%
# Average the predicted risk by job role.
display(
    utils.summarize_risk_by(
        test_df, ag_proba, "JobRole", label, positive_class=positive_class
    )
)
# Outcome: sales representatives have the highest predicted and observed
# attrition.

# %% [markdown]
# ## Cell 6.2: Limits of This Model
#
# - The data is fictional and small
#   - Use it to learn the workflow, not to draw conclusions about real employees
# - The features include `Age`, `Gender`, and `MaritalStatus`
#   - A real HR model must be checked for unfair treatment of protected groups
#     before anyone acts on its scores
# - The model ranks risk: it does not say why people leave or what would keep
#   them
#   - Answering that needs causal analysis or experiments

# %% [markdown]
# ## Summary: The Mental Model
#
# - AutoGluon turns a raw table into a validated ensemble in one `fit()` call, but
#   the choices around it stay with you
#   - Remove identifiers, pick the metric, and keep a simple baseline
# - On small tables, report confidence intervals: the gap between models is often
#   smaller than the noise of the test set
# - The model gives probabilities, the decision needs a threshold
#   - Choose it on out-of-fold predictions with a metric or a cost that reflects
#     the business, then store it with `set_decision_threshold()`
# - Permutation importance explains what the model uses overall, SHAP explains
#   single predictions; neither one explains causes
