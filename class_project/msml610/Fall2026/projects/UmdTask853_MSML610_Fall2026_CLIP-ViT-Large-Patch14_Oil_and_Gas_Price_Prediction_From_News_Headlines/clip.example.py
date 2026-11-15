# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Oil & natural gas return forecasting — frozen CLIP benchmark
#
# **Research objective.** Test whether frozen CLIP text representations from energy news improve next-observation log-return RMSE over zero, train-mean, and price-only Ridge baselines.
#
# The experiment uses chronological train/validation/test splits with five forecast-origin observations omitted around boundaries. All scalers and PCA transforms are fitted **within** expanding-window training folds. Headlines with date-only timestamps are conservatively assigned to a forecast origin strictly later than their publication date. This is *not* a deployable real-time trading strategy.
#
# **Required local inputs:** official FRED CSV exports `DCOILWTICO.csv`, `DHHNGSP.csv`; optionally the ABC Australia `abcnews-date-text.csv`. The notebook deliberately does not fabricate benchmark numbers when those files are absent.

# %% [markdown]
# ## 1. Configuration and data availability

# %%
from pathlib import Path
import pandas as pd

project = Path.cwd()
wti_csv = project / "data" / "DCOILWTICO.csv"
gas_csv = project / "data" / "DHHNGSP.csv"
news_csv = project / "data" / "abcnews-date-text.csv"
results = project / "results" / "benchmark"

print("WTI CSV present:", wti_csv.exists())
print("Gas CSV present:", gas_csv.exists())
print("ABC headlines present:", news_csv.exists())
if not (wti_csv.exists() and gas_csv.exists()):
    print("Download the two official FRED CSV exports into data/ (see BENCHMARK_README.md).")

# %% [markdown]
# ## 2. Price-only benchmark
#
# Use the real FRED data to create next-observation returns and fit zero, mean, and expanding-window-tuned Ridge baselines. Missing/nonpositive observations invalidate adjacent log returns.

# %%
import subprocess
import sys

if wti_csv.exists() and gas_csv.exists():
    command = [sys.executable, "run_forecasting.py", "--wti-csv", str(wti_csv),
               "--gas-csv", str(gas_csv), "--output-dir", str(results)]
    subprocess.run(command, check=True)
    display(pd.read_csv(results / "summary.csv"))
else:
    print("Price benchmark was not run because FRED price CSVs are missing.")

# %% [markdown]
# ## 3. Frozen CLIP headline features and controlled comparison
#
# With ABC headlines, encode each headline with frozen `openai/clip-vit-large-patch14`, average embeddings for each available forecast date, and compare price-only Ridge against price + 20 PCA components, price + zero-shot event scores, and their combination. A shuffled-news negative control breaks date-content associations. CLIP can lose to price-only Ridge; that is a valid finding.

# %%
if wti_csv.exists() and gas_csv.exists() and news_csv.exists():
    command = [sys.executable, "run_forecasting.py", "--wti-csv", str(wti_csv),
               "--gas-csv", str(gas_csv), "--headlines-csv", str(news_csv),
               "--output-dir", str(results), "--batch-size", "16"]
    subprocess.run(command, check=True)
    display(pd.read_csv(results / "summary.csv"))
else:
    print("CLIP experiment not run. Supply FRED prices and the ABC news CSV first.")

# %% [markdown]
# ## 4. Interpret cumulative squared errors on unseen test observations
#
# Cumulative error curves show *when* a model gained or lost ground. They should be interpreted alongside aggregate RMSE and the block-bootstrap confidence intervals.

# %%
if (results / "wti_predictions.csv").exists():
    from IPython.display import Image
    plot = results / "wti_cumulative_test_error.png"
    subprocess.run([sys.executable, "plot_forecast_errors.py", "--predictions",
                    str(results / "wti_predictions.csv"), "--output", str(plot)], check=True)
    display(Image(filename=str(plot)))
else:
    print("Run the benchmark to produce test prediction files before plotting.")

# %% [markdown]
# ## 5. Methodological checklist
#
# - **Forecast horizon:** predicting the following recorded commodity-price observation, not necessarily the next calendar day.
# - **Leakage:** no shuffled or timestamped headline can use the forecast target; preprocessing stays inside time-series CV folds.
# - **Source coverage:** ABC News has sparse and geographically biased energy coverage; check `has_news` coverage in the console output and run an explicit news-availability audit.
# - **2020 WTI anomaly:** skip undefined log returns touching nonpositive spot prices, rather than silently bridging over them.
# - **Reporting:** report improvements *and* failures; a 95% CI spanning zero is not evidence of a reliable RMSE gain. The bootstrap CI is conditional on fitted forecasts.
# - **Further research:** test event novelty and market regimes only after the required baseline and negative-control experiments are reproducible.
