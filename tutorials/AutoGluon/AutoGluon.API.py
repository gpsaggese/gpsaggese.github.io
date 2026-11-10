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
# # AutoGluon API Overview
#
# - This notebook explores `autogluon.tabular` on a synthetic customer-churn table
#   whose label depends on known features, so every output can be checked against
#   the truth
# - The pedagogical arc:
#   - Load a table and let AutoGluon infer the problem and the feature types
#   - Train a zoo of models with one `fit()` call and read the leaderboard
#   - Predict, evaluate, and explain the predictor
#   - Control the training: model families, bagging, stacking, and presets
#   - Reuse the same API for regression and multiclass problems
#   - Save, reload, and slim down a predictor for deployment

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import autogluon.tabular as agt
import autogluon.tabular.configs.presets_configs as agpresets
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
# ## Library Overview
#
# - **What problem it solves**: training a strong model on a table takes many
#   steps
#   - Encode categorical, datetime, and text columns
#   - Pick and tune several model families
#   - Validate without leaking the test set
#   - Ensemble the best models
#   - AutoGluon runs all of them from one call, in a time budget you set
# - **Key abstraction**: a `TabularPredictor` owns one label and one directory
#   - `fit()` fills the directory with a set of trained models and an ensemble
#   - Every other method reads from it
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `TabularDataset(data)` | A `pandas.DataFrame` subclass | Also loads a CSV or Parquet file from a path or URL |
# | `TabularPredictor(label, eval_metric, path)` | Container of the trained models | Infers `problem_type` from the label column |
# | `.fit(train_data, presets, time_limit)` | Trains many models, then a weighted ensemble | `presets` trades accuracy against time, size, and latency |
# | `.leaderboard(data)` | One row per trained model | Scores, fit time, and prediction time |
# | `.predict(data)` / `.predict_proba(data)` | Predictions of the best model | The best model is often the ensemble |
# | `.evaluate(data)` | Dictionary of metrics | Every score is "higher is better" |
# | `.feature_importance(data)` | Permutation importance per feature | Drop of the score when a column is shuffled |
#
# - **Key classes**:
#   - `TabularPredictor`: fits, stores, and serves the models
#   - `TabularDataset`: reads tables, behaves like a `DataFrame`

# %% [markdown]
# # Part 1: Data

# %% [markdown]
# ## Cell 1.1: Generate a Table With a Known Answer
#
# - The generator returns one row per customer with mixed column types
#   - Numbers, strings, a date, missing values, and a constant column
# - `churned` depends only on `contract`, `support_calls`, `tenure_months`, and
#   `monthly_charges`
#   - `region`, `signup_date`, and `country` are noise by construction
# - Two more columns can act as labels later:
#   - `lifetime_value` for regression
#   - `contract` for multiclass classification

# %%
# Generate 2,000 synthetic customers.
customers_df = utils.make_synthetic_customers(2000)
print("customers_df.shape=", customers_df.shape)
display(customers_df.head())

# %%
# Count the missing values per column.
display(customers_df.isna().sum().to_frame("n_missing"))
# Outcome: only `monthly_charges` has missing values, about 5% of the rows.

# %% [markdown]
# ## Cell 1.2: Split Train and Test
#
# - The split is stratified on `churned`, so both sets have the same churn rate
# - `lifetime_value` is removed from the classification data
#   - It would be a feature here, and a label only in Part 5

# %%
# Split the data, keeping the churn rate equal in the two sets.
train_df, test_df = utils.split_train_test(customers_df, stratify_col="churned")
churn_train_df = train_df.drop(columns=["lifetime_value"])
churn_test_df = test_df.drop(columns=["lifetime_value"])
print("churn_train_df.shape=", churn_train_df.shape)
print("churn_test_df.shape=", churn_test_df.shape)
print(
    "churn rate (train)=", round((churn_train_df["churned"] == "yes").mean(), 3)
)

# %% [markdown]
# ## Cell 1.3: `TabularDataset` Is a `DataFrame`
#
# - `TabularDataset` adds file loading to `pandas.DataFrame` and nothing else
#   - E.g., `agt.TabularDataset("https://.../train.csv")` downloads and parses a file
# - Every `TabularPredictor` method also accepts a plain `DataFrame`

# %%
# Wrap the training data and check its type.
train_data = agt.TabularDataset(churn_train_df)
print("type(train_data)=", type(train_data))
print(
    "isinstance(train_data, pd.DataFrame)=",
    isinstance(train_data, pd.DataFrame),
)

# %% [markdown]
# # Part 2: Train a Predictor

# %% [markdown]
# ## Cell 2.1: Fit With One Call
#
# - `TabularPredictor` needs only the label
#   - `eval_metric` defaults to `accuracy` for classification
#   - ROC-AUC is used here since it scores the ranking of the probabilities and
#     ignores the 0.5 threshold
#   - `path` is the directory where the models are saved
# - `fit()` arguments used here:
#   - `presets="medium_quality"`: fast training, no bagging (see Part 4)
#   - `time_limit`: upper bound in seconds for the whole fit
#   - `excluded_model_types`: skips the neural networks, which need `torch`, not
#     installed in this image
# - The log prints each decision: problem type, feature processing, models
#   trained, and the validation score of each one

# %%
# Train the default model zoo on the churn data.
predictor = agt.TabularPredictor(
    label="churned",
    eval_metric="roc_auc",
    path="tmp.autogluon/api_churn",
)
_ = predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=60,
    excluded_model_types=["NN_TORCH", "FASTAI"],
)

# %% [markdown]
# ## Cell 2.2: Read What AutoGluon Inferred
#
# - The label has two values, so the problem is `binary`
# - The positive class is the second value in sorted order (`yes`)
#   - Pass `positive_class=...` to `TabularPredictor` to choose it
# - `fit()` held out part of the training data to validate the models, since the
#   preset does not bag

# %%
# Print the problem definition inferred from the label column.
print("predictor.problem_type=", predictor.problem_type)
print("predictor.eval_metric=", predictor.eval_metric.name)
print("predictor.class_labels=", predictor.class_labels)
print("predictor.positive_class=", predictor.positive_class)

# %% [markdown]
# ## Cell 2.3: Inspect the Feature Types
#
# - AutoGluon assigns a type to each column, then a feature generator per type
#   - Strings with few values become `category`
#   - The date becomes an integer and 4 calendar columns
# - Compare the input columns and the features the models use
#   - `country` has a single value, so it is dropped

# %%
# Show the type and the special tags of each feature.
print(predictor.feature_metadata)

# %%
# Compare the original columns with the features the models use.
print("predictor.original_features=", predictor.original_features)
print("predictor.features()=", predictor.features())

# %%
# Apply the same feature processing to new rows.
display(predictor.transform_features(churn_test_df.head(3)))
# Outcome: `signup_date` is expanded in year, month, day, and day of week.

# %% [markdown]
# # Part 3: Predict and Evaluate

# %% [markdown]
# ## Cell 3.1: Predict Labels and Probabilities
#
# - `predict()` returns a label per row
#   - For a binary problem the label is "positive" when the probability is at
#     least `predictor.decision_threshold`, 0.5 by default
# - `predict_proba()` returns one column per class
# - The test rows still contain `churned`: AutoGluon ignores the label column at
#   prediction time

# %%
# Predict the label of 5 test customers.
display(predictor.predict(churn_test_df.head()).to_frame())

# %%
# Predict the probability of each class for the same customers.
display(predictor.predict_proba(churn_test_df.head()))

# %% [markdown]
# ## Cell 3.2: Evaluate on the Test Set
#
# - `evaluate()` returns the eval metric and other metrics for the problem type
# - The binary metrics except ROC-AUC depend on the 0.5 threshold
#   - `AutoGluon.example.ipynb` shows how to move the threshold

# %%
# Compute the test metrics of the best model.
test_scores = predictor.evaluate(churn_test_df)
display(pd.Series(test_scores, name="score").to_frame())

# %% [markdown]
# # Part 4: Inspect and Control the Models

# %% [markdown]
# ## Cell 4.1: Read the Leaderboard
#
# - One row per model
#   - `score_val`: score on the internal validation data, used to rank the models
#     and build the ensemble
#   - `score_test`: score on the data passed to `leaderboard()`
#   - `fit_time` and `pred_time_test`: the cost of each model, in seconds
# - `WeightedEnsemble_L2` averages the predictions of the other models with
#   weights learned on the validation data

# %%
# Score every model on the test set.
leaderboard_df = predictor.leaderboard(churn_test_df)
cols = ["model", "score_test", "score_val", "fit_time", "pred_time_test"]
display(leaderboard_df[cols])

# %%
# Plot the test score of each model.
utils.plot_leaderboard(leaderboard_df, title="Test ROC-AUC per model")
# Outcome: most models are within a few points of each other, since the
# generator is a simple logistic model.

# %% [markdown]
# ## Cell 4.2: Explain the Predictor With Feature Importance
#
# - `feature_importance()` shuffles one column at a time and measures the drop of
#   the eval metric
#   - It explains the whole predictor, ensemble included
#   - `num_shuffle_sets` repeats the shuffle to estimate the noise (`stddev`,
#     `p_value`)
# - The generator gives the right answer: `region` and `signup_date` are noise

# %%
# Compute the permutation importance of each feature on the test set.
importance_df = predictor.feature_importance(churn_test_df, num_shuffle_sets=3)
display(importance_df)

# %%
# Plot the importances with one standard deviation.
utils.plot_feature_importance(importance_df)
# Outcome: `contract`, `support_calls`, and `tenure_months` lead, while `region`
# and `signup_date` are close to 0.

# %% [markdown]
# ## Cell 4.3: Choose the Models, Then Bag and Stack Them
#
# - `hyperparameters` maps a model family to its settings
#   - `{}` means the default settings
#   - E.g., `GBM` is LightGBM, `CAT` is CatBoost, `XT` is Extra Trees
# - `num_bag_folds=5` trains each model on 5 folds (bagging)
#   - Every row gets an out-of-fold prediction, so no data is held out
# - `num_stack_levels=1` adds a second layer of models that also read the
#   out-of-fold predictions of the first layer (stacking)
# - The model names show the structure
#   - `_BAG_L1` is a bagged first-layer model
#   - `_BAG_L2` is a bagged second-layer model
# - The image does not install `ray`, so the folds are trained one after the
#   other, as the log says

# %% [markdown]
# - `verbosity=1` hides the training log from now on

# %%
# Train three model families with 5-fold bagging and one stack level.
bagged_predictor = agt.TabularPredictor(
    label="churned",
    eval_metric="roc_auc",
    path="tmp.autogluon/api_churn_bagged",
    verbosity=1,
)
_ = bagged_predictor.fit(
    train_data,
    hyperparameters={"GBM": {}, "CAT": {}, "XT": {}},
    num_bag_folds=5,
    num_stack_levels=1,
    time_limit=120,
)

# %%
# Show the layers of the bagged and stacked predictor.
bagged_leaderboard_df = bagged_predictor.leaderboard(churn_test_df)
cols = ["model", "score_test", "score_val", "stack_level", "fit_time"]
display(bagged_leaderboard_df[cols])
# Outcome: the second layer does not beat `CatBoost_BAG_L1`, so the ensembles
# put all their weight on it. Stacking helps on larger and harder tables, not on
# a simple logistic generator.

# %%
# Print the weight of each model in the second-layer ensemble.
ensemble_info = bagged_predictor.info()["model_info"]["WeightedEnsemble_L2"]
print("weights=", ensemble_info["children_info"]["S1F1"]["model_weights"])

# %% [markdown]
# ## Cell 4.4: Presets Bundle These Choices
#
# - A preset is a dictionary of `fit()` arguments
#   - `medium_quality`: no bagging, the default
#   - `good_quality` and `high_quality`: bagging, stacking, then a refit of the
#     best models on all the data (`refit_full`) for fast inference
#   - `best_quality`: keeps all the bagged models for the highest accuracy and the
#     largest model
# - Arguments passed to `fit()` override the preset, e.g., `time_limit`

# %%
# Show the `fit()` arguments set by each common preset.
presets = ["medium_quality", "good_quality", "high_quality", "best_quality"]
presets_dict = agpresets.tabular_presets_dict
presets_df = pd.DataFrame(
    {name: pd.Series(presets_dict[name]) for name in presets}
).T
display(presets_df)

# %% [markdown]
# # Part 5: Other Problem Types

# %% [markdown]
# ## Cell 5.1: Regression
#
# - The same code with a float label trains regressors
#   - The default eval metric becomes the root mean squared error (RMSE)
# - AutoGluon reports error metrics with a minus sign, so that a higher score is
#   always better
#   - An RMSE of 350 shows up as `-350`

# %%
# Predict the customer lifetime value from the other columns.
value_train_df = train_df.drop(columns=["churned"])
value_test_df = test_df.drop(columns=["churned"])
regressor = agt.TabularPredictor(
    label="lifetime_value",
    path="tmp.autogluon/api_value",
    verbosity=1,
)
_ = regressor.fit(
    value_train_df, hyperparameters={"GBM": {}, "LR": {}}, time_limit=30
)
print("regressor.problem_type=", regressor.problem_type)
print("regressor.eval_metric=", regressor.eval_metric.name)

# %%
# Evaluate the regressor on the test set.
display(pd.Series(regressor.evaluate(value_test_df), name="score").to_frame())
# Outcome: R2 is close to 1, since the value is tenure times charges plus 10%
# noise.

# %% [markdown]
# ## Cell 5.2: Multiclass Classification
#
# - A string label with more than two values gives a `multiclass` problem
# - `predict_proba()` returns one column per class

# %%
# Predict the contract type of each customer.
contract_predictor = agt.TabularPredictor(
    label="contract",
    path="tmp.autogluon/api_contract",
    verbosity=1,
)
_ = contract_predictor.fit(
    churn_train_df, hyperparameters={"GBM": {}}, time_limit=30
)
print("contract_predictor.problem_type=", contract_predictor.problem_type)
print("contract_predictor.class_labels=", contract_predictor.class_labels)
display(contract_predictor.predict_proba(churn_test_df.head(3)))

# %% [markdown]
# # Part 6: Save, Load, and Deploy

# %% [markdown]
# ## Cell 6.1: Reload a Predictor From Its Directory
#
# - `fit()` saves everything in `predictor.path` as it goes
# - `TabularPredictor.load(path)` restores the predictor in a new process
#   - The AutoGluon and Python versions must match the ones used to train

# %%
# Reload the first predictor and check that it gives the same predictions.
reloaded = agt.TabularPredictor.load(predictor.path)
are_equal = reloaded.predict(churn_test_df).equals(
    predictor.predict(churn_test_df)
)
print("predictor.path=", predictor.path)
print("are_equal=", are_equal)

# %% [markdown]
# ## Cell 6.2: Keep Only What Prediction Needs
#
# - `clone_for_deployment()` copies the predictor without the models that the
#   best model does not use
# - The clone predicts the same values, from a smaller directory
#   - Here the best model is the ensemble, which only uses `CatBoost_BAG_L1`
#     (Cell 4.3), so the clone keeps 2 models

# %%
# Clone the bagged predictor for deployment and compare the sizes on disk.
deploy_path = bagged_predictor.clone_for_deployment(
    "tmp.autogluon/api_churn_deploy", dirs_exist_ok=True
)
deployed = agt.TabularPredictor.load(deploy_path)
sizes_df = pd.DataFrame(
    {
        "n_models": [
            len(bagged_predictor.model_names()),
            len(deployed.model_names()),
        ],
        "disk_usage_mb": [
            round(bagged_predictor.disk_usage() / 1e6, 1),
            round(deployed.disk_usage() / 1e6, 1),
        ],
    },
    index=["bagged_predictor", "deployed"],
)
display(sizes_df)

# %% [markdown]
# ## Summary: The Mental Model
#
# - A `TabularPredictor` is a directory of models built from one label
#   - `fit()` infers the problem, processes the features, trains several model
#     families, and ensembles them within `time_limit`
# - The leaderboard and `feature_importance()` show what was trained and what it
#   uses
#   - Check them against what you know about the data before trusting a score
# - Presets are bundles of `fit()` arguments
#   - Start with `medium_quality` to iterate
#   - Move to `good_quality` or `best_quality` once the data is right
# - The same five calls cover binary, multiclass, and regression problems
#   - `fit()`, `predict()`, `predict_proba()`, `evaluate()`, `leaderboard()`
