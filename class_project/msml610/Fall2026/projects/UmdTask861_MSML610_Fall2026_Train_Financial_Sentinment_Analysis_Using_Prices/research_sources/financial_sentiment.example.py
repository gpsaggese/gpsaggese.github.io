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
# # Financial sentiment using price labels: audit example
#
# This notebook follows the actual retained data from article-stock candidates to eligible exploratory outcomes. It checks exclusions and independently recomputes stock-minus-SPY returns.
#
# **This is an outcome audit, not model evaluation or a trading backtest.**

# %%
import json
import pandas as pd
import financial_sentiment_utils as utils

audit = utils.load_audit()
print(f"Model: {utils.MODEL_STATUS}; evaluation: {utils.EVALUATION_STATUS}")

# %% [markdown]
# ## Measure actual coverage
#
# Each record type is counted separately. Multiple ticker tags on one article create several candidate rows, not several independent pieces of information.

# %%
print(pd.DataFrame(utils.coverage_rows(audit)).to_string(index=False))

# %% [markdown]
# ## Check candidate eligibility
#
# The saved audit contains 203 article-stock candidates and 90 available labels across 81 unique articles. The remaining rows are excluded under the audit rules. These expectations describe this snapshot, not a target accuracy.

# %%
labels = pd.DataFrame(audit["labels"])
print(labels.groupby(["symbol", "status"]).size().rename("rows").to_string())
print("\nSaved eligibility reasons are preserved; they are not sentiment classes.")

# %% [markdown]
# ## Recompute the 60-minute outcomes
#
# Assume availability at the later of publication and revision, wait 60 seconds, then use the next five-minute bar open. Compare the stock return with SPY over the same 60 minutes. Require the same regular-session date and an exit no later than 15:55.
#
# The assumed availability does not prove that the archived text version was actually received then. No spread, fees, slippage, executable quote, or portfolio decision is modeled.

# %%
verification = utils.verify_exploratory_labels(audit)
print(json.dumps(verification, indent=2))
assert verification["candidate_article_stock_pairs"] == 203
assert verification["available_exploratory_labels"] == 90
assert verification["unique_articles_with_labels"] == 81

# %% [markdown]
# ## Inspect a small outcome sample
#
# Returns below are decimal realized labels. They are not predicted probabilities, sentiment annotations, trade recommendations, or an estimate of model performance.

# %%
available = labels.loc[labels["status"] == "exploratory_label_available"]
columns = ["event_id", "symbol", "entry_at", "exit_at", "stock_return", "spy_return", "excess_return"]
print(available[columns].head(5).to_string(index=False))

# %% [markdown]
# ## What would count as the next result?
#
# Collect persistent first-seen records; build a sentiment baseline and a price-label text model; evaluate on later dates; then compare Quant.ai variants at equal initial capital and risk constraints. None of those results exists in this starter.
#
# Template reference: [UMD project template, commit 60df5bc](https://github.com/gpsaggese/umd_classes/tree/60df5bc966d6da403eb54ee059372edc5cf20099/class_project/project_template).
