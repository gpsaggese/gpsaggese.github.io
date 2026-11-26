# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # House Price Prediction Example
#
# This notebook demonstrates the end-to-end machine learning workflow for
# the Kaggle House Prices regression task without requiring a running server.
#
# - Loads (or generates) the dataset via `template_utils`.
# - Compares multiple sklearn regression models using cross-validation.
# - Trains the best model on the full dataset and saves it to `ml_model/`.
# - Runs direct in-process predictions using the saved model.
# - Produces a neighbourhood price comparison chart saved to `results/`.
# - Reference: (house_price.example.md)
#
# Follow the reference to write notebooks in a clear manner:
# https://github.com/causify-ai/helpers/blob/master/docs/tools/notebooks/all.jupyter_notebook.how_to_guide.md

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# ## Imports

# %%
import logging
import os
import sys

sys.path.insert(0, "/project")

import template_utils as cpptteut

# %% [markdown]
# ## Configuration

# %%
_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Load data
#
# Attempt to load the Kaggle CSV; fall back to a synthetic dataset if the
# file is absent so the notebook runs without Kaggle credentials.

# %%
# Load or generate the House Prices dataset.
DATA_PATH = "ml/train.csv"
df = cpptteut.load_data(DATA_PATH)
_LOG.info("Dataset shape: %s", df.shape)
df.head()

# %% [markdown]
# ## Compute stats
#
# Inspect the raw data before any cleaning to understand distributions and
# identify potential issues.

# %%
# Display summary statistics for the target and key numeric features.
print("Target column statistics:")
print(df[cpptteut.TARGET_COLUMN].describe())
print(
    f"\nMissing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}"
)

# %% [markdown]
# ## Split data

# %%
# Split into train and test sets for offline evaluation.
X_train, X_test, y_train, y_test = cpptteut.split_data(df)
_LOG.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))

# %% [markdown]
# ## Compare models
#
# Cross-validate GradientBoosting, RandomForest, and Ridge and display the
# leaderboard sorted by RMSE.

# %%
# Compare all candidate models using 5-fold cross-validation.
leaderboard = cpptteut.compare_models(df, fold=5)
leaderboard

# %% [markdown]
# ## Train best model

# %%
# Train the top-ranked model (GradientBoosting) on the full dataset.
best_model_name = leaderboard.iloc[0]["Model"]
_LOG.info("Best model: %s", best_model_name)
pipeline = cpptteut.train_best_model(df, model_name=best_model_name)

# %% [markdown]
# ## Evaluate model

# %%
# Evaluate the fitted pipeline on the held-out test set.
metrics = cpptteut.evaluate_model(pipeline, X_test, y_test)
print(f"Test RMSE : ${metrics['RMSE']:,.0f}")
print(f"Test MAE  : ${metrics['MAE']:,.0f}")
print(f"Test R²   : {metrics['R2']:.4f}")

# %% [markdown]
# ## Save model

# %%
# Persist the trained pipeline to disk for the Flask API to load.
cpptteut.finalize_and_save(pipeline)
_LOG.info("Model saved.")

# %% [markdown]
# ## Run in-process predictions

# %%
# Load the saved artifact and run a single in-process prediction.
model = cpptteut.load_model_artifact()
house = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "YearBuilt": 2005,
    "Neighborhood": "CollgCr",
}
price = cpptteut.predict_price(house, model=model)
_LOG.info("Predicted price: $%.0f", price)
print(f"Predicted sale price: ${price:,.0f}")

# %% [markdown]
# ## Validate features

# %%
# Demonstrate validation with an intentionally bad payload.
bad_payload = {"OverallQual": 15, "GrLivArea": -50, "ExterQual": "ZZ"}
errors = cpptteut.validate_features(bad_payload)
print("Validation errors:")
for e in errors:
    print(f"  ✗ {e}")

# %% [markdown]
# ## Show results
#
# Compare predicted prices across neighbourhoods and save the chart.

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Build one instance per neighbourhood using default feature values.
neighborhoods = ["OldTown", "BrkSide", "CollgCr", "NWAmes", "NoRidge"]
instances = [
    {**cpptteut.FEATURE_DEFAULTS, "Neighborhood": n} for n in neighborhoods
]
# Predict prices for all neighbourhoods.
prices = [cpptteut.predict_price(inst, model=model) for inst in instances]
result_df = pd.DataFrame(
    {"Neighborhood": neighborhoods, "PredictedPrice": prices}
).sort_values("PredictedPrice")
# Plot and save the neighbourhood comparison chart.
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.barh(result_df["Neighborhood"], result_df["PredictedPrice"] / 1000)
plt.xlabel("Predicted Price ($k)")
plt.title("Predicted Price by Neighbourhood (median feature house)")
plt.tight_layout()
plt.savefig("results/price_by_neighborhood.png", dpi=120)
plt.show()
_LOG.info("Plot saved to results/price_by_neighborhood.png.")
print(result_df.to_string(index=False))

# %%
