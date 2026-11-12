# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Metalearners
#
# - This notebook estimates heterogeneous treatment effects (CATE) with
#   metalearners, and compares them on email marketing and discount data with
#   gain curves
# - The pedagogical arc:
#   - Loading a biased (observational) and a randomized email marketing
#     dataset
#   - T-Learner: separate outcome models per treatment arm
#   - X-Learner: second-stage models with propensity-score weighting
#   - S-Learner: a single model with treatment as a covariate, and
#     counterfactual predictions
#   - Double ML / R-Learner: debiasing and denoising nuisance models

# %%
# `lightgbm`/`fklearn` are extras for this notebook, not in the shared
# requirements.txt, so they are self-installed here (see `utils` for the
# `fklearn` usage).
import helpers.hmodule as hmodule

hmodule.install_module_if_not_present(
    ["lightgbm", "fklearn"],
    use_activate=True,
    use_sudo=False,
    venv_path="/opt/venv",
)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from lightgbm import LGBMRegressor

# %%
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

import L08_04_07_metalearners_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: Loading Data

# %% [markdown]
# ## Cell 1.1: Loading the email marketing datasets
#
# **Goal**
# - Load a biased (observational) and a randomized email marketing
#   dataset, sharing the same schema, used throughout the T/X-Learner
#   sections

# %%
dir_name = "L08_data"
# !ls $dir_name

out_dir_name = "figures/"

# %%
data_biased = pd.read_csv(f"{dir_name}/email_obs_data.csv")
print("# data_biased")
print("num_rows=", len(data_biased))
display(data_biased.head())

data_rnd = pd.read_csv(f"{dir_name}/email_rnd_data.csv")
print("# data_rnd")
print("num_rows=", len(data_rnd))
display(data_rnd.head())

# %%
hdbg.dassert_eq(data_biased.columns.tolist(), data_rnd.columns.tolist())

# %%
y = "next_mnth_pv"
T = "mkt_email"
X = list(data_rnd.drop(columns=[y, T]).columns)

train, test = data_biased, data_rnd

display(train[[T, y]].head())

# %% [markdown]
# # Part 2: T-Learner

# %% [markdown]
# ## Cell 2.1: Fitting separate outcome models per treatment arm
#
# **Goal**
# - Fit one outcome model on the control arm and one on the treated arm,
#   the T-Learner's defining idea, then predict the CATE as their
#   difference
#
# **Implementation**
# - `m0`/`m1` are `LGBMRegressor` models fit on `T==0`/`T==1` rows
#   respectively; `cate = m1.predict(x) - m0.predict(x)`

# %%
np.random.seed(123)

m0 = LGBMRegressor()
m1 = LGBMRegressor()

m0.fit(train.query(f"{T}==0")[X].values, train.query(f"{T}==0")[y].values)
_ = m1.fit(train.query(f"{T}==1")[X].values, train.query(f"{T}==1")[y].values)

display(m0)

# %% [markdown]
# ## Cell 2.2: Evaluating the T-Learner CATE with a gain curve
#
# **Goal**
# - Check the T-Learner's CATE estimate on the held-out randomized data,
#   via a cumulative gain curve
#
# **Implementation** `utils.plot_gain_curve_analysis(cate_test, T, y, ...)`

# %%
t_learner_cate_test = test.assign(
    cate=m1.predict(test[X].values) - m0.predict(test[X].values)
)

_ = utils.plot_gain_curve_analysis(t_learner_cate_test, T, y, title="T-Learner")

# %% [markdown]
# ## Cell 2.3: Visualizing treatment-effect heterogeneity on synthetic data
#
# **Goal**
# - Fit the same T-Learner pattern on synthetic data with a known,
#   heterogeneous treatment effect, to visualize what the 2 outcome
#   models and the resulting CATE actually look like
#
# **Implementation** `utils.generate_synthetic_treatment_data(...)`,
# `utils.fit_tlearner_models(...)`,
# `utils.plot_tlearner_treatment_effect_analysis(...)`

# %%
# Generate synthetic data with treatment heterogeneity.
df = utils.generate_synthetic_treatment_data(n0=500, n1=50, seed=123)

# Fit separate outcome models for control and treatment groups.
m0, m1, m0_hat, m1_hat = utils.fit_tlearner_models(df, min_child_samples=25)

# %%
# Visualize outcome models and heterogeneous treatment effects.
utils.plot_tlearner_treatment_effect_analysis(df, m0, m1, m0_hat, m1_hat)

# %% [markdown]
# # Part 3: X-Learner

# %% [markdown]
# ## Cell 3.1: Estimating heterogeneous treatment effects
#
# **Goal**
# - Compute the X-Learner's imputed treatment effects: for each unit, the
#   difference between its observed outcome and the *other* arm's
#   predicted outcome
#
# **Implementation**
# - `utils.calculate_xlearner_heterogeneous_treatment_effects(df, m0, m1)`

# %%
tau_0, tau_1 = utils.calculate_xlearner_heterogeneous_treatment_effects(
    df, m0, m1
)

# %% [markdown]
# ## Cell 3.2: Fitting X-Learner second-stage models on synthetic data
#
# **Goal**
# - Fit a second-stage model on each arm's imputed effects, then compare
#   the resulting CATE estimates against the ground truth
#
# **Implementation** `utils.fit_xlearner_models(df, tau_0, tau_1, ...)`,
# `utils.plot_xlearner_effect_estimates(...)`

# %%
mu_tau0, mu_tau1, mu_tau0_hat, mu_tau1_hat = utils.fit_xlearner_models(
    df, tau_0, tau_1, min_child_samples=25
)

# %%
# Plot heterogeneous treatment effect estimates.
utils.plot_xlearner_effect_estimates(df, tau_0, tau_1, mu_tau0_hat, mu_tau1_hat)

# %% [markdown]
# ## Cell 3.3: Visualizing propensity-score-weighted CATE
#
# **Goal**
# - Combine the 2 second-stage estimates into a single CATE, weighted by
#   the propensity score, and visualize the result
#
# **Implementation** `utils.plot_xlearner_with_propensity_scores(...)`

# %%
utils.plot_xlearner_with_propensity_scores(df, mu_tau0, mu_tau1, tau_0, tau_1)

# %% [markdown]
# ## Cell 3.4: Fitting the propensity-score-weighted X-Learner on real data
#
# **Goal**
# - Repeat the X-Learner pattern on the real email data: first-stage
#   models weighted by the inverse propensity score, then second-stage
#   models on the residual treatment effects
#
# **Implementation**
# - `utils.fit_propensity_score_and_weighted_outcome_models(train, X, T, y)`
# - `utils.fit_xlearner_second_stage_models(train, X, T, y, m0, m1)`

# %%
# Fit propensity score model and first-stage outcome models with inverse
# probability weighting.
ps_model, m0, m1 = utils.fit_propensity_score_and_weighted_outcome_models(
    train, X, T, y
)

# %%
# Fit second-stage models on residual treatment effects.
m_tau_0, m_tau_1 = utils.fit_xlearner_second_stage_models(train, X, T, y, m0, m1)

# %% [markdown]
# ## Cell 3.5: Evaluating the X-Learner CATE with a gain curve
#
# **Goal**
# - Check the X-Learner's CATE estimate on the held-out data, via the
#   same cumulative gain curve used for the T-Learner

# %%
x_cate_test = utils.estimate_xlearner_cate(test, X, ps_model, m_tau_0, m_tau_1)

_ = utils.plot_gain_curve_analysis(x_cate_test, T, y, title="X-Learner")

# %% [markdown]
# # Part 4: S-Learner

# %% [markdown]
# ## Cell 4.1: Loading the discount dataset
#
# **Goal**
# - Load a continuous-treatment dataset (discount level, not a binary
#   flag), split chronologically into train/test

# %%
data_cont = pd.read_csv(f"{dir_name}/discount_data.csv")
print("data_cont.shape[0]=", data_cont.shape[0])
display(data_cont.head())

# %%
train = data_cont.query("day<'2018-01-01'")
print("train.shape[0]=", train.shape[0])
test = data_cont.query("day>='2018-01-01'")
print("test.shape[0]=", test.shape[0])

# %% [markdown]
# ## Cell 4.2: Fitting a single shared-covariate model
#
# **Goal**
# - Fit one model with treatment as a plain input feature (the S-Learner
#   pattern), instead of 2 separate per-arm models
#
# **Implementation** `utils.fit_slearner_model(train, X, T, y)`

# %%
X = ["month", "weekday", "is_holiday", "competitors_price"]
T = "discounts"
y = "sales"

s_learner = utils.fit_slearner_model(train, X, T, y)

# %% [markdown]
# ## Cell 4.3: Generating counterfactual predictions
#
# **Goal**
# - Sweep the discount level through a grid of counterfactual values, to
#   see how predicted sales respond
#
# **Implementation**
# - `utils.generate_slearner_counterfactual_predictions(test, X, T, y,
#   s_learner, discount_grid)`

# %%
test_cf = utils.generate_slearner_counterfactual_predictions(
    test, X, T, y, s_learner, np.array([0, 10, 20, 30, 40])
)

display(test_cf.head(8))

# %% [markdown]
# ## Cell 4.4: Visualizing counterfactual sales curves
#
# **Goal**
# - Plot the predicted sales-vs-discount curve for a few selected days,
#   for one restaurant, to see how the shape varies day to day

# %%
days = ["2018-12-25", "2018-01-01", "2018-06-01", "2018-06-18"]

plt.figure(figsize=(10, 4))
_ = sns.lineplot(
    data=test_cf.query("day.isin(@days)").query("rest_id==2"),
    y="sales_hat",
    x="discounts",
    style="day",
)

# %% [markdown]
# ## Cell 4.5: Evaluating the S-Learner CATE with a gain curve
#
# **Goal**
# - Estimate the CATE from the counterfactual predictions, then check it
#   with the same cumulative gain curve used for the T/X-Learners
#
# **Implementation** `utils.estimate_slearner_cate(test_cf, test, T, y)`

# %%
test_s_learner_pred = utils.estimate_slearner_cate(test_cf, test, T, y)

display(test_s_learner_pred.head())

# %%
_ = utils.plot_gain_curve_analysis(test_s_learner_pred, T, y, title="S-Learner")

# %% [markdown]
# # Part 5: Double ML / R-Learner

# %% [markdown]
# ## Cell 5.1: Fitting the debiasing and denoising nuisance models
#
# **Goal**
# - Fit the 2 nuisance models the R-Learner needs: one predicting
#   treatment from covariates (debiasing), one predicting outcome from
#   covariates alone (denoising)
#
# **Implementation** `utils.fit_rlearner_models(train, X, T, y)`

# %%
X = ["month", "weekday", "is_holiday", "competitors_price"]
T = "discounts"
y = "sales"

# Fit debiasing and denoising models.
debias_m, denoise_m = utils.fit_rlearner_models(train, X, T, y)

# %% [markdown]
# ## Cell 5.2: Fitting the R-Learner CATE model on residuals
#
# **Goal**
# - Regress the outcome residual on the treatment residual (both from the
#   nuisance models above), whose slope is the R-Learner's CATE estimate
#
# **Implementation** `utils.fit_rlearner_cate_model(train, X, T, y,
# debias_m, denoise_m)`

# %%
cate_model, t_res, y_res = utils.fit_rlearner_cate_model(
    train, X, T, y, debias_m, denoise_m
)

# %% [markdown]
# ## Cell 5.3: Checking coefficients via an OLS diagnostic
#
# **Goal**
# - Cross-check the R-Learner's residual-on-residual regression against a
#   plain OLS fit, as an optional diagnostic

# %%
display(sm.OLS(y_res, t_res).fit().summary().tables[1])

# %% [markdown]
# ## Cell 5.4: Evaluating the R-Learner CATE with a gain curve
#
# **Goal**
# - Estimate the CATE on the test set, then check it with the same
#   cumulative gain curve used for the other 3 metalearners

# %%
test_r_learner_pred = utils.estimate_rlearner_cate(test, X, cate_model)

_ = utils.plot_gain_curve_analysis(test_r_learner_pred, T, y, title="R-Learner")
