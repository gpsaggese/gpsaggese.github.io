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
# # Factor Investing: Two Procedures to Test a Factor
#
# - This notebook runs the two procedures that factor researchers use to test
#   whether a characteristic explains differences in average returns across
#   securities, on a simulated universe where the true premium is known
# - The pedagogical arc:
#   - Cross-section versus time-series variation
#   - The model $Y = X \beta + Z \gamma + \varepsilon$ on the data of one period
#   - Procedure 1 (Fama-MacBeth): one cross-sectional regression per period,
#     then the mean and standard deviation of the estimates
#   - Procedure 2 (Fama-French): quantile portfolios, a long-short portfolio,
#     and time-series regressions
#   - Statistical significance of $\beta$, and a comparison of the two
#     procedures

# %%
# !pip install -q statsmodels==0.14.6

import statsmodels

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import L07_01_factor_investing_utils as utils

print("statsmodels version: ", statsmodels.__version__)

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
# # Part 1: The Cross-Section and the Model
#
# - Every cell simulates a universe with `utils.simulate_universe()`
#   - $N$ securities observed over $T$ periods
#   - Characteristics $x_{i,k,t}$, standardized in each period and persistent
#     over time for each security ($k$ indexes factors, e.g., value, size)
#   - A control variable $z_{i,t}$, correlated with $x_{i,1,t}$ through `rho`
# - The return of security $i$ in period $t$ depends on the characteristics
#   observed at $t-1$
#   - $r_{i,t} = m_t + s_{i,t} + \varepsilon_{i,t}$
#   - $s_{i,t} = \sum_k \lambda_{k,t} x_{i,k,t-1} + \gamma z_{i,t-1}$
#   - $m_t$: common market return, shared by all securities
#   - $\lambda_{k,t} \sim N(\mu_k, \tau^2)$: factor premium, with mean $\mu_k$
#     and $\tau = 0.5$
#   - $\varepsilon_{i,t} \sim N(0, \sigma^2)$: idiosyncratic noise
# - Returns are in percent per period
# - The true $\mu_k$ is known, so every estimate can be checked against it

# %%
hintros.print_obj_info(utils.simulate_universe)

# %% [markdown]
# ## Cell 1.1: Cross-Section Versus Time-Series Variation
#
# **Goal**
# - Separate the two sources of variation in returns:
#   - Differences across securities
#   - Changes over time
# - Both procedures in this notebook explain the first one, not the second
# - See that a characteristic can explain the cross-section of average returns
#   even when it explains little of the total return variance

# %% [markdown]
# **Description**
# - Inputs
#   - `mu`: true premium per unit of characteristic, 0.0-1.0 percent per period
#   - `sigma`: idiosyncratic volatility, 1-10 percent per period
#   - `log2(N)`: number of securities $N$, 32-1024 on a log scale
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Cross-section`: one dot per security
#     - x-axis: time-averaged characteristic
#     - y-axis: time-averaged return
#     - Solid line: least-squares fit, dotted line: true slope $\mu$
#   - `Time series`: cross-sectional mean return in each period, the common
#     movement the procedures do not try to explain
#   - `Variance split`: shares of the return variance from the common time
#     effect, the characteristic, and the idiosyncratic noise
#   - `Comments`: current parameters, fitted slope, $R^2$ of the cross-section,
#     and the variance shares

# %%
# Compare the cross-section of average returns with the time series.
utils.cell1_1_cross_section_vs_time_series()

# %% [markdown]
# **Guided usage**
# - Drag `mu` from 0.0 to 1.0, leaving `sigma` fixed
#   - Observe the `Cross-section` scatter tilt into an upward line, and $R^2$
#     rise
#   - Observe `Time series` stay unchanged: the cross-sectional mean return does
#     not depend on $\mu$
#   - Observe the characteristic share in `Variance split` stay at a few
#     percent: the factor explains average returns, not most of the variance
# - Raise `sigma` from 2 to 10
#   - Observe the dots spread around the line and $R^2$ fall, although the true
#     `mu` is the same
# - Change `seed` a few times at $N = 32$, then at $N = 1024$
#   - Observe the fitted slope jump around `mu` for small $N$, and stay closer
#     to it for large $N$

# %% [markdown]
# **Implementation** `cell1_1_cross_section_vs_time_series(figsize=None)`
# - Simulates $T = 120$ periods with `simulate_universe()`
# - Averages each security over time for `Cross-section`, and fits the line
#   with `np.polyfit()`
# - Averages each period over securities for `Time series`
# - Splits the variance using the true components of the simulation:
#   - The market $m_t$
#   - The characteristic term $\lambda_t x_{i,t-1}$
#   - The noise $\varepsilon_{i,t}$

# %%
hintros.print_obj_info(utils.cell1_1_cross_section_vs_time_series)

# %% [markdown]
# ## Cell 1.2: Anatomy of Returns, Factors, and Controls
#
# **Goal**
# - Name the pieces of $Y = X \beta + Z \gamma + \varepsilon$ on the data of one
#   period:
#   - Returns $Y$
#   - Explanatory factors $X$
#   - Control variables $Z$
# - Build on Cell 1.1: the cross-section there is now split into its columns

# %% [markdown]
# **Description**
# - Inputs
#   - `t`: period shown, 1-120
#   - `K`: number of factors, 1-3, the columns of `X`
#   - `N`: number of securities shown, 8-64, the rows
#   - `sort`: order the rows by the first characteristic $x_1$, highest on top
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Y`: heat strip of the returns of the $N$ securities in period `t`
#   - `X`: heatmap of the $K$ standardized characteristics, one column per
#     factor
#   - `Z`: heat strip of the standardized control variable
#   - `Comments`: current parameters, true $\mu_k$, $\gamma$, and $\sigma$, and
#     the correlation between each column of `X` and `Y`

# %%
# Show the returns, factors, and controls of one period.
utils.cell1_2_returns_factors_controls()

# %% [markdown]
# **Guided usage**
# - Step `t` from 1 to 10
#   - Observe `Y` change at every step while `X` and `Z` drift slowly: the
#     characteristics are persistent, the returns are not
# - Raise `K` from 1 to 3
#   - Observe a new column appear in `X` and a new correlation line in
#     `Comments` for each factor
#   - Observe $x_3$ correlate negatively with `Y` in most periods, since
#     $\mu_3 < 0$
# - Check `sort`
#   - Observe `Y` shade from warm at the top to cool at the bottom in most
#     periods: the pattern the regressions of Part 2 measure

# %% [markdown]
# **Implementation** `cell1_2_returns_factors_controls(figsize=None)`
# - Simulates a low-noise universe with `simulate_universe()`, so the pattern
#   is visible:
#   - $\sigma = 1$ and $\mu = (1.0, 0.5, -0.5)$
#   - $\gamma = 0.5$ and a correlation of 0.3 between $x_1$ and $z$
# - Slices period `t` into three dataframes, optionally sorted by $x_1$
# - Draws each one with `sns.heatmap()`, centered at zero for `X` and `Z`, and
#   at the period mean for `Y`

# %%
hintros.print_obj_info(utils.cell1_2_returns_factors_controls)

# %% [markdown]
# # Part 2: Procedure 1, Cross-Sectional Regressions

# %% [markdown]
# ## Cell 2.1: One-Period Cross-Sectional Regression
#
# **Goal**
# - Estimate the factor exposure $\hat\beta_t$ of one period by regressing the
#   returns of all securities on their characteristic
#   - $r_{i,t} = \alpha_t + \beta_t x_{i,t-1} + e_{i,t}$
# - See that one period gives a noisy estimate, which motivates the averaging of
#   Cell 2.2

# %% [markdown]
# **Description**
# - Inputs
#   - `t`: period whose cross-section is regressed, 1-120
#   - `log2(N)`: number of securities $N$, 32-1024 on a log scale
#   - `sigma`: idiosyncratic volatility, 1-10 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Scatter`: returns against the characteristic in period `t`, with three
#     lines:
#     - Solid: least-squares fit with slope $\hat\beta_t$
#     - Dotted: true line with slope $\lambda_t$
#     - Dashed: reference line with the mean premium $\mu = 0.5$
#   - `Estimates`: $\hat\beta_t$ with its 95 percent interval, next to the true
#     $\lambda_t$ and $\mu$
#   - `Residuals`: histogram of the fitted residuals
#   - `Comments`: current parameters, $\hat\beta_t$ with its standard error and
#     interval, $R^2$, the intercept, and the true values

# %%
# Regress the returns of one period on the characteristic.
utils.cell2_1_one_period_regression()

# %% [markdown]
# **Guided usage**
# - Raise `log2(N)` from 5 to 10 ($N$ from 32 to 1024) at `t` = 1
#   - Observe the interval in `Estimates` narrow, from about $\pm 1.8$ to about
#     $\pm 0.3$
#   - Observe $\hat\beta_t$ move toward the true $\lambda_t$, not toward $\mu$:
#     one period measures the premium of that period
# - Step `t` from 1 to 10 at $N = 128$
#   - Observe $\hat\beta_t$ jump around the dashed $\mu$ line, and fall below
#     zero in some periods: a single period is unreliable
# - Raise `sigma` from 2 to 10
#   - Observe the interval widen and $R^2$ fall

# %% [markdown]
# **Implementation** `cell2_1_one_period_regression(figsize=None)`
# - Simulates the universe with `simulate_universe()`, with $\mu = 0.5$ and
#   $T = 120$
# - Fits $r_{i,t}$ on a constant and $x_{i,t-1}$ with `sm.OLS()`
# - Reads the slope, its standard error, and its interval from `params`, `bse`,
#   and `conf_int()`

# %%
hintros.print_obj_info(utils.cell2_1_one_period_regression)

# %% [markdown]
# ## Cell 2.2: Averaging Betas Across Periods
#
# **Goal**
# - Follow the Fama-MacBeth procedure end to end:
#   - One cross-sectional regression per period, as in Cell 2.1
#   - The mean $\bar\beta$ and standard deviation $s_\beta$ of $\hat\beta_t$
#     across all periods
# - Turn the mean and the standard deviation into a $t$-statistic for the
#   premium
#   - $t = \bar\beta / (s_\beta / \sqrt{T})$

# %% [markdown]
# **Description**
# - Inputs
#   - `log2(T)`: number of periods $T$, 16-512 on a log scale
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `sigma`: idiosyncratic volatility, 1-10 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Beta path`: $\hat\beta_t$ over the periods, with the sample mean (solid)
#     and the true $\mu$ (dotted)
#   - `Histogram`: distribution of $\hat\beta_t$, with the mean and plus or
#     minus one standard deviation marked
#   - `Running mean`: running mean of $\hat\beta_t$, with a band of plus or minus
#     two standard errors
#   - `Comments`: current parameters, mean and standard deviation of
#     $\hat\beta_t$, standard error $s_\beta / \sqrt{T}$, and the $t$-statistic

# %%
# Run the Fama-MacBeth procedure and test the mean premium.
utils.cell2_2_fama_macbeth()

# %% [markdown]
# **Guided usage**
# - Set `mu` to 0.2 and raise `log2(T)` from 4 to 9 ($T$ from 16 to 512)
#   - Observe the `Running mean` band narrow, and the $t$-statistic grow roughly
#     with $\sqrt{T}$, from about 1 to about 8
# - Set `mu` to 0.0 and step `seed` from 0 to 20 at $T = 128$
#   - Observe the $t$-statistic stay inside $\pm 2$ for most seeds but leave it
#     for seed 9: a premium that does not exist can look significant
# - Set `mu` back to 0.5 and raise `sigma` from 2 to 10
#   - Observe `Beta path` get noisier and the $t$-statistic fall, while the mean
#     stays near `mu`

# %% [markdown]
# **Implementation** `cell2_2_fama_macbeth(figsize=None)`
# - Simulates $N = 128$ securities with `simulate_universe()`
# - Runs one `sm.OLS()` fit per period in `run_fama_macbeth()`
# - Summarizes $\hat\beta_t$ with `summarize_estimates()`:
#   - Mean and standard deviation
#   - Standard error and $t$-statistic
# - Computes the running mean and its band with `expanding()`

# %%
hintros.print_obj_info(utils.cell2_2_fama_macbeth)

# %% [markdown]
# ## Cell 2.3: Controls Remove Omitted-Variable Bias
#
# **Goal**
# - Show why the regression includes control variables $Z$: leaving out a
#   control that is correlated with the factor biases $\hat\beta$
#   - The bias is about $\gamma \rho$ for standardized variables
# - Build on Cell 2.2 by running the same procedure twice, without and with $Z$

# %% [markdown]
# **Description**
# - Inputs
#   - `rho`: correlation between the factor $x$ and the control $z$, 0.0-0.9
#   - `gamma`: effect of the control on returns, 0.0-1.0 percent per period
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `X versus Z`: characteristic against control in the last period
#   - `Estimates`: mean $\hat\beta$ without and with the control, each with its
#     95 percent interval, next to the true $\mu$
#   - `Beta path`: $\hat\beta_t$ of both specifications over the periods
#   - `Comments`: current parameters, and the mean, bias, standard error, and
#     $t$-statistic of each specification

# %%
# Compare Fama-MacBeth estimates with and without the control.
utils.cell2_3_controls_and_bias()

# %% [markdown]
# **Guided usage**
# - Keep `gamma` at 0.5 and raise `rho` from 0.0 to 0.8
#   - Observe the `without Z` bar drift above $\mu$ by about
#     $\gamma \rho = 0.4$, while the `with Z` interval still covers $\mu$
#   - Observe the `with Z` interval widen: a correlated control costs precision
# - Set `gamma` to 0.0 with `rho` at 0.8
#   - Observe both intervals cover $\mu$ again: omitting a control matters only
#     if the control drives returns
# - Set `rho` to 0.0 with `gamma` at 0.5
#   - Observe both bars agree: an uncorrelated control does not bias the factor

# %% [markdown]
# **Implementation** `cell2_3_controls_and_bias(figsize=None)`
# - Simulates a control $z$ with `simulate_universe()`:
#   - Correlated with $x$ through `rho`
#   - Loading on returns through `gamma`
# - Calls `run_fama_macbeth()` twice:
#   - On $x$ alone
#   - With `use_control=True`, on $x$ and $z$
# - Summarizes both series of $\hat\beta_t$ with `summarize_estimates()`

# %%
hintros.print_obj_info(utils.cell2_3_controls_and_bias)

# %% [markdown]
# # Part 3: Procedure 2, Sorted Portfolios

# %% [markdown]
# ## Cell 3.1: Ranking Securities Into Quantile Portfolios
#
# **Goal**
# - Build quantile portfolios from a characteristic:
#   - Rank the securities by the characteristic
#   - Partition the universe into quantile subsets
#   - Compute the time series of average returns of each subset
# - See the value idea directly: top-ranked securities outperform bottom-ranked
#   ones when the premium is real

# %% [markdown]
# **Description**
# - Inputs
#   - `q`: number of quantile portfolios, 2-10
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `sigma`: idiosyncratic volatility, 1-10 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Ranking`: securities of the last period sorted by their characteristic,
#     colored by quantile
#   - `Average return`: mean excess return of each quantile portfolio, with a 95
#     percent interval
#     - Excess return: return minus the universe mean of the same period, which
#       removes the common market move
#   - `Cumulative return`: cumulative excess return of each quantile portfolio
#   - `Comments`: current parameters, securities per quantile, and the mean
#     return of the top and the bottom quantile with their spread

# %%
# Sort securities into quantile portfolios and compare their returns.
utils.cell3_1_quantile_portfolios()

# %% [markdown]
# **Guided usage**
# - Keep `mu` at 0.5 and `q` at 5
#   - Observe the `Average return` bars step up from Q1 to Q5
#   - Observe the `Cumulative return` lines fan out in the same order
# - Set `mu` to 0.0
#   - Observe every bar stay within its interval of zero, and the lines stay
#     close to zero with no consistent order
# - Set `mu` back to 0.5 and raise `q` from 3 to 10
#   - Observe the top-minus-bottom spread widen, from about 1.1 to about 1.8
#   - Observe the intervals of the extreme quantiles widen too: each portfolio
#     holds fewer securities

# %% [markdown]
# **Implementation** `cell3_1_quantile_portfolios(figsize=None)`
# - Simulates $N = 256$ securities over $T = 120$ periods with
#   `simulate_universe()`
# - Converts the panel to one row per period and security with
#   `to_long_frame()`
# - Assigns the quantiles of the characteristic known at the start of each
#   period with `assign_quantiles()`, a `groupby().rank(pct=True)` per period
# - Averages the excess returns per period and quantile with `groupby()`, then
#   over time

# %%
hintros.print_obj_info(utils.cell3_1_quantile_portfolios)

# %% [markdown]
# ## Cell 3.2: Building the Long-Short Portfolio
#
# **Goal**
# - Turn the ranking of Cell 3.1 into one return series that isolates the
#   factor:
#   - Positive weights on top-ranked securities
#   - Negative weights on bottom-ranked securities
# - See how the choice of legs and weights changes the factor return series

# %% [markdown]
# **Description**
# - Inputs
#   - `frac`: share of securities in each leg, 0.1-0.5
#   - `scheme`: equal weights within each leg, or rank weights that grow toward
#     the extremes
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Weights`: weight of each security in the last period against its rank
#     - Green: long leg, summing to +1
#     - Red: short leg, summing to -1
#   - `Long-short return`: bars of the return in each period, with the
#     cumulative return as a line on the right axis
#   - `Distribution`: histogram of the long-short returns, with the mean marked
#   - `Comments`: current parameters, the sum of each leg, and the mean,
#     standard deviation, and $t$-statistic of the long-short return

# %%
# Build the long-short portfolio and show its return series.
utils.cell3_2_long_short_portfolio()

# %% [markdown]
# **Guided usage**
# - Raise `frac` from 0.1 to 0.5
#   - Observe `Weights` fill the whole range of ranks, with no zero-weight
#     middle
#   - Observe the mean long-short return fall, from about 1.8 to about 0.8: the
#     legs now include less extreme securities
#   - Observe the standard deviation fall too, from about 2.2 to about 1.0:
#     each leg holds more securities
# - Switch `scheme` from equal to rank at `frac` = 0.3
#   - Observe `Weights` change from flat steps to a ramp that favors the
#     extremes
# - Set `mu` to 0.0
#   - Observe the `Distribution` center near zero, and the cumulative line
#     wander without a trend

# %% [markdown]
# **Implementation** `cell3_2_long_short_portfolio(figsize=None)`
# - Ranks the characteristic in each period and selects the two legs in
#   `long_short_weights()`
# - Scales the long leg to +1 and the short leg to -1, so the market return
#   cancels out
# - Computes the portfolio return as the weighted sum of the security returns,
#   and summarizes it with `summarize_estimates()`

# %%
hintros.print_obj_info(utils.cell3_2_long_short_portfolio)

# %% [markdown]
# ## Cell 3.3: Time-Series Regression on Factor Returns
#
# **Goal**
# - Regress the return series $Y$ of each quintile portfolio on the long-short
#   factor return series $X$, plus an optional control series $Z$
#   - $Y_t = a + \beta X_t + c Z_t + e_t$
# - Show that $\hat\beta$ differs across portfolios and lines up with their
#   average returns: the factor explains differences in average returns

# %% [markdown]
# **Description**
# - Inputs
#   - `quantile`: quintile portfolio shown in `Scatter` and outlined elsewhere,
#     Q1-Q5
#   - `control`: no control, or the market return as $Z$
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Scatter`: return of the selected portfolio against the long-short return,
#     one dot per period, with a line of slope $\hat\beta$
#   - `Beta by quantile`: $\hat\beta$ of each quintile portfolio with a 95
#     percent interval
#   - `Mean return versus beta`: one dot per quintile portfolio, average return
#     against $\hat\beta$, with a fitted line
#   - `Comments`: current parameters, and the $\hat\beta$, $t$-statistic,
#     intercept, and $R^2$ of the selected portfolio

# %%
# Regress each quintile portfolio on the long-short factor returns.
utils.cell3_3_time_series_regression()

# %% [markdown]
# **Guided usage**
# - Step `quantile` from Q1 to Q5
#   - Observe the `Scatter` slope turn from negative to positive
#   - Observe the outlined point climb the line in `Mean return versus beta`
# - Switch `control` from none to market
#   - Observe each $\hat\beta$ move little, but its interval shrink from about
#     $\pm 0.56$ to below $\pm 0.1$
#   - Observe $R^2$ jump from a few percent to about 0.98: the market return
#     explains most of the variation that the factor does not
# - Set `mu` to 0.0
#   - Observe `Beta by quantile` stay the same: $\hat\beta$ measures comovement
#     with the factor, which does not depend on the mean premium
#   - Observe the line in `Mean return versus beta` go flat: with no premium, a
#     higher $\hat\beta$ earns no higher average return

# %% [markdown]
# **Implementation** `cell3_3_time_series_regression(figsize=None)`
# - Builds the three series:
#   - $Y$: quintile portfolios of raw returns, with `assign_quantiles()` and
#     `groupby()`
#   - $X$: equal-weighted long-short returns with `frac = 0.3`, from
#     `long_short_weights()`
#   - $Z$: the equal-weighted mean return of the universe
# - Fits one `sm.OLS()` per quintile, and reads `params`, `bse`, `tvalues`, and
#   `rsquared`
# - Fits the line of `Mean return versus beta` with `np.polyfit()`

# %%
hintros.print_obj_info(utils.cell3_3_time_series_regression)

# %% [markdown]
# ## Cell 3.4: Multifactor Sorts Partition Each Subset
#
# **Goal**
# - Extend the sort to two factors, value $x_1$ and size $x_2$:
#   - Partition by the first factor
#   - Partition each subset again by the second factor
# - See what a second factor adds to the portfolio returns, and when the order
#   of the sorts matters, building on Cell 3.1

# %% [markdown]
# **Description**
# - Inputs
#   - `q`: number of quantiles per factor, 2-5
#   - `sort`: how the second factor is ranked
#     - Sequential: within each $x_1$ subset
#     - Independent: on the full universe
#   - `rho`: correlation between the two characteristics, 0.0-0.9
#   - `mu1`: premium of the first factor (value), 0.0-1.0 percent per period
#   - `mu2`: premium of the second factor (size), 0.0-1.0 percent per period
#   - `seed`: random seed, redraws the whole universe
#
# - Panels
#   - `Mean return grid`: mean excess return of each cell of the `q` by `q` sort,
#     $x_1$ quantile on rows and $x_2$ quantile on columns
#   - `Count grid`: average number of securities in each cell
#   - `Marginal means`: mean return by $x_1$ quantile and by $x_2$ quantile,
#     each averaged over the other factor
#   - `Comments`: current parameters, the smallest and the largest cell count,
#     and the mean-return range along each factor

# %%
# Show double-sorted portfolios on two characteristics.
utils.cell3_4_multifactor_sort()

# %% [markdown]
# **Guided usage**
# - Set `mu2` to 0.0, with `mu1` at 0.5 and `rho` at 0.0
#   - Observe `Mean return grid` vary down the rows only
#   - Observe the `by x2 quantile` line in `Marginal means` go flat
# - Keep `sort` on independent and raise `rho` from 0.0 to 0.9
#   - Observe `Count grid` collapse onto the diagonal, with empty corner cells:
#     when the characteristics move together, few securities rank high on one
#     and low on the other
# - Switch `sort` to sequential at `rho` = 0.9
#   - Observe `Count grid` become uniform, with 40 securities per cell: every
#     $x_1$ subset is partitioned in full

# %% [markdown]
# **Implementation** `cell3_4_multifactor_sort(figsize=None)`
# - Simulates $N = 1000$ securities over $T = 60$ periods, with two factors
#   whose correlation is `rho`
# - Assigns the $x_1$ quantiles per period with `assign_quantiles()`
# - Assigns the $x_2$ quantiles:
#   - Per period, for the independent sort
#   - Per period and $x_1$ quantile, for the sequential sort
# - Averages the excess returns and counts the securities of each cell with
#   `groupby()`, in `cell3_4_compute_double_sort()`

# %%
hintros.print_obj_info(utils.cell3_4_multifactor_sort)

# %% [markdown]
# # Part 4: Significance and Comparison

# %% [markdown]
# ## Cell 4.1: Statistical Significance and False Discoveries
#
# **Goal**
# - Understand what "$\hat\beta$ is statistically significant" means for a
#   researcher stating a case: a Fama-MacBeth $t$-statistic beyond $\pm 2$
# - See how often a factor with no premium still passes the test, and how the
#   chance of detecting a real premium depends on $T$, building on Cell 2.2

# %% [markdown]
# **Description**
# - Inputs
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `log2(T)`: number of periods $T$ per study, 16-512 on a log scale
#   - `log2(n_trials)`: number of simulated studies, 32-1024 on a log scale
#   - `seed`: random seed for all the studies
#
# - Panels
#   - `t-stat distribution`: histogram of the $t$-statistics of all the studies,
#     with the rejection zones $|t| > 2$ shaded
#   - `Rejection rate`: share of studies with $|t| > 2$ against the true $\mu$,
#     with a marker at the current `mu` and a dotted line at 5 percent
#   - `Comments`: current parameters, the share of studies with $|t| > 2$, and
#     the mean $t$-statistic

# %%
# Show how often studies find a significant premium.
utils.cell4_1_significance_and_false_discoveries()

# %% [markdown]
# **Guided usage**
# - Keep `mu` at 0.0 and raise `log2(n_trials)` to 10 (1024 studies)
#   - Observe about 5 percent of the studies land in the rejection zones,
#     although no premium exists
# - Raise `mu` from 0.0 to 0.2 at $T = 64$
#   - Observe the histogram shift right, and the marker climb to about 65
#     percent on the `Rejection rate` curve
# - Set `mu` to 0.1 and raise `log2(T)` from 4 to 8 ($T$ from 16 to 256)
#   - Observe the share of significant studies rise from about 10 to about 65
#     percent: a small premium is detectable only with a long sample

# %% [markdown]
# **Implementation** `cell4_1_significance_and_false_discoveries(figsize=None)`
# - Draws the Fama-MacBeth slope of each period from its exact sampling
#   distribution, in `cell4_1_simulate_studies()`
#   - $\hat\beta_t = \lambda_t + e_t$, with $\lambda_t \sim N(\mu, \tau^2)$ and
#     $e_t \sim N(0, \sigma^2 / N)$
#   - This holds for a characteristic standardized in each period and Gaussian
#     noise, and avoids running $T$ regressions per study
# - Computes the $t$-statistic of each study, and the rejection rate on a grid
#   of $\mu$ values with the same draws

# %%
hintros.print_obj_info(utils.cell4_1_significance_and_false_discoveries)

# %% [markdown]
# ## Cell 4.2: Comparing the Two Procedures
#
# **Goal**
# - Run both procedures on the same simulated studies, and compare the premium
#   estimates and the verdicts
# - Close the arc: both procedures target differences in average returns across
#   securities, and reach similar conclusions by different routes

# %% [markdown]
# **Description**
# - Inputs
#   - `mu`: true mean premium, 0.0-1.0 percent per period
#   - `log2(T)`: number of periods $T$ per study, 16-512 on a log scale
#   - `log2(N)`: number of securities $N$, 32-1024 on a log scale
#   - `frac`: share of securities in each long-short leg, 0.1-0.5
#   - `seed`: random seed for all the studies
#
# - Panels
#   - `Estimates`: mean Fama-MacBeth $\hat\beta$ and mean rescaled long-short
#     return of the first study, each with a 95 percent interval, next to the
#     true $\mu$
#     - The long-short return is divided by the gap $s$ in mean characteristic
#       between the legs, so both are in percent per unit of characteristic
#   - `t-stat scatter`: one dot per study, Fama-MacBeth $t$-statistic against
#     long-short $t$-statistic, with the 45 degree line and the $\pm 2$
#     thresholds
#   - `Verdicts`: share of the 500 studies significant under both procedures,
#     only Fama-MacBeth, only long-short, and neither
#   - `Comments`: current parameters, the correlation between the two
#     $t$-statistics, and the share of studies where the verdicts disagree

# %%
# Compare the verdicts of the two procedures on the same studies.
utils.cell4_2_compare_procedures()

# %% [markdown]
# **Guided usage**
# - Raise `mu` from 0.0 to 0.3 at $T = 64$
#   - Observe the cloud in `t-stat scatter` move up and right, and the `both`
#     bar grow from about 4 to about 90 percent
#   - Observe the correlation between the two $t$-statistics stay near 0.95
# - Set `mu` to 0.15, where the premium is borderline
#   - Observe the `only FM` and `only LS` bars reach about 10 percent together:
#     near the threshold, small differences flip the verdict
# - Raise `log2(N)` from 5 to 10 ($N$ from 32 to 1024)
#   - Observe the cloud tighten around the 45 degree line, with the correlation
#     rising from about 0.92 to about 0.99: with many securities, both estimates
#     follow the same premium path $\lambda_t$
# - Move `frac` from 0.3 to 0.1 or to 0.5
#   - Observe the correlation drop to about 0.89 and the disagreement grow: in
#     this simulation, legs of 30 percent track the regression best

# %% [markdown]
# **Implementation** `cell4_2_compare_procedures(figsize=None)`
# - Draws the estimates of both procedures in each period from their exact
#   joint sampling distribution, in `cell4_2_simulate_both_procedures()`
#   - Fama-MacBeth: $\hat\beta_t = \lambda_t + e_F$
#   - Long-short: $L_t = \lambda_t s + e_L$, rescaled to $L_t / s$
#   - $e_F$ and $e_L$ come from the same noise, so they are correlated
# - Computes the $t$-statistic of each study under both procedures
# - Classifies each study by verdict with `np.select()`, and counts the shares
#   with `value_counts()`

# %%
hintros.print_obj_info(utils.cell4_2_compare_procedures)

# %%
# TODO(gp): Use available data
