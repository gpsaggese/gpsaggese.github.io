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
# # House Price Prediction API
#
# This notebook demonstrates how to interact with the Flask REST API that
# serves predictions from a scikit-learn regression model trained on the
# Kaggle House Prices dataset.
#
# - Covers the `/health`, `/features`, `/predict`, and `/predict/batch` endpoints.
# - Uses `template_utils` helpers to keep API calls clean and reusable.
# - Requires the API server to be running: `python app.py`
# - Reference: (house_price.API.md)
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

import requests
import template_utils as cpptteut

# %% [markdown]
# ## Configuration

# %%
_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Check API health
#
# Verify the server is running and the model is loaded before making
# prediction requests.

# %%
# Call the health endpoint and display the server status.
health = cpptteut.api_health()
_LOG.info("API health: %s", health)
print(health)

# %% [markdown]
# ## Inspect available features
#
# Retrieve the full feature catalogue and default values so we know what
# fields we can pass to /predict.

# %%
# Fetch the feature catalogue from the API.
resp = requests.get(f"{cpptteut.DEFAULT_API_URL}/features")
features = resp.json()
# Display numeric and categorical feature names with their defaults.
print("Numeric features:")
for f in features["numeric_features"]:
    print(f"  {f:<20s}  default={features['defaults'].get(f)}")
print("\nCategorical features:")
for f in features["categorical_features"]:
    print(f"  {f:<20s}  default={features['defaults'].get(f)}")

# %% [markdown]
# ## Single prediction
#
# Send one house's features to /predict and display the result.

# %%
# Build the request payload with a subset of features.
# Missing fields will be filled with server-side defaults.
payload = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "YearBuilt": 2005,
    "Neighborhood": "CollgCr",
    "ExterQual": "Gd",
    "KitchenQual": "Gd",
}
# Post the payload and display the predicted sale price.
result = cpptteut.api_predict(payload)
_LOG.info("Predicted price: %s", result["predicted_price"])
print(f"Predicted sale price: ${result['predicted_price']:,.0f}")

# %% [markdown]
# ## Batch prediction
#
# Send multiple houses in one request to /predict/batch and compare
# predicted prices across different quality and size combinations.

# %%
# Define a batch of houses varying quality and living area.
instances = [
    {"OverallQual": 3, "GrLivArea": 800},
    {"OverallQual": 5, "GrLivArea": 1200},
    {"OverallQual": 7, "GrLivArea": 1800},
    {"OverallQual": 9, "GrLivArea": 3000},
]
# Post the batch request and display a comparison table.
batch = cpptteut.api_predict_batch(instances)
print(f"{'Quality':>10}  {'Area (sqft)':>12}  {'Predicted Price':>16}")
print("-" * 44)
for inst, price in zip(instances, batch["predictions"]):
    print(
        f"{inst['OverallQual']:>10}  {inst['GrLivArea']:>12,}  ${price:>15,.0f}"
    )

# %% [markdown]
# ## Price sensitivity analysis
#
# Hold all features at their default values and vary OverallQual from 1
# to 10 to visualise how quality drives price.

# %%
import matplotlib.pyplot as plt

# Build instances across the full quality range.
qual_range = list(range(1, 11))
instances = [{"OverallQual": q, "GrLivArea": 1500} for q in qual_range]
# Fetch batch predictions for all quality levels.
prices = cpptteut.api_predict_batch(instances)["predictions"]
# Plot price vs quality.
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(qual_range, [p / 1000 for p in prices], marker="o", linewidth=2)
plt.xlabel("Overall Quality (1–10)")
plt.ylabel("Predicted Price ($k)")
plt.title("Predicted Sale Price vs. Overall Quality  (GrLivArea = 1 500 sqft)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/price_vs_quality.png", dpi=120)
plt.show()
_LOG.info("Plot saved to results/price_vs_quality.png.")

# %%
