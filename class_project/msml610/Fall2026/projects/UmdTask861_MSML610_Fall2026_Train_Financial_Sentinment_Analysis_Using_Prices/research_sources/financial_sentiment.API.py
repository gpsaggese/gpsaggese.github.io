# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Financial sentiment using price labels: data API
#
# This notebook inspects saved Quant.ai data without network calls. The sequence is: load the archive, explain fields, inspect bounded real samples, and state what the data cannot establish.
#
# **Model: Not trained. Evaluation: Not evaluated.**

# %%
import pandas as pd
import financial_sentiment_utils as utils

print(utils.PROJECT_TITLE)
print(f"Model: {utils.MODEL_STATUS}; evaluation: {utils.EVALUATION_STATUS}")

# %% [markdown]
# ## Load the retained audit
#
# The default directory is resolved relative to the utility module, not the current working directory. Set `FINANCIAL_SENTIMENT_DATA_DIR` to use an existing archive elsewhere. Missing files fail explicitly; no source is contacted.

# %%
audit = utils.load_audit()
print("Archive:", audit["data_dir"])
print(pd.DataFrame(utils.coverage_rows(audit)).to_string(index=False))

# %% [markdown]
# ## Field meanings
#
# Publication, update, and receipt times have different meanings. Historical text and price outcomes must never be passed to a model as if they were known in advance.

# %%
print(pd.DataFrame(utils.field_dictionary()).to_string(index=False))

# %% [markdown]
# ## Inspect real source records
#
# Only short titles and metadata are shown. Vendor ticker tags require a relevance review. Futu text remains in its original language and has not received a validated sentiment label.

# %%
for source in ("news", "futu"):
    print("\nSource:", source)
    print(pd.DataFrame(utils.sample_records(audit, source, limit=2)).to_string(index=False))

# %% [markdown]
# ## Interpretation limits
#
# No author-level contrarian strategy can be evaluated from the Futu snapshot. A body or excerpt is not guaranteed full text. The archive has no proven historical news receipt trail. This API exploration does not train, score, or trade a model.
#
# Template reference: [UMD project template, commit 60df5bc](https://github.com/gpsaggese/umd_classes/tree/60df5bc966d6da403eb54ee059372edc5cf20099/class_project/project_template). This is a self-contained Quant.ai adaptation.
