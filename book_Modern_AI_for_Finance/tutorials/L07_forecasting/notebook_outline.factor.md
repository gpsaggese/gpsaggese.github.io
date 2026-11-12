# Notebook Overview: Factor Investing Procedures

- Scope
  - Interactive tour of the two procedures factor researchers use to test whether a
    characteristic explains differences in average returns across securities
  - Both procedures run on a simulated universe, so the true factor premium is known
    and every estimate can be checked against it
- The pedagogical arc:
  - Cross-section versus time-series variation
  - The model $Y = X \beta + Z \gamma + \varepsilon$ on actual data
  - Procedure 1 (Fama-MacBeth): one cross-sectional regression per period, then the
    mean and standard deviation of the estimates
  - Procedure 2 (Fama-French): quantile portfolios, a long-short portfolio, and
    time-series regressions
  - Statistical significance of $\beta$, and a comparison of the two procedures
- Shared simulation, built once in the paired `*_utils.py` and reused by every cell
  - Universe of $N$ securities observed over $T$ periods
  - Characteristics $x_{i,k,t}$ with $k$ indexing factors (e.g., value, size)
    - Standardized in each period
    - Persistent over time for each security
  - Control variable $z_{i,t}$ correlated with $x_{i,1,t}$ through `rho`
  - Return of security $i$ in period $t$ depends on characteristics observed at
    $t-1$
    - $r_{i,t} = m_t + s_{i,t} + \varepsilon_{i,t}$
    - $s_{i,t} = \sum_k \lambda_{k,t} x_{i,k,t-1} + \gamma z_{i,t-1}$
    - $m_t$: common market return, shared by all securities
    - $\lambda_{k,t} \sim N(\mu_k, \tau^2)$: factor premium, with mean $\mu_k$
    - $\varepsilon_{i,t} \sim N(0, \sigma^2)$: idiosyncratic noise
  - $\tau$, the persistence of the characteristics, and the volatility of $m_t$ are
    fixed defaults in the utils, not widgets
- Packages
  - `statsmodels` for least-squares fits and standard errors, installed with a pinned
    `!pip install` cell if it is not in the lesson `requirements.txt`
  - All other cells use the shared base: `ipywidgets`, `matplotlib`, `numpy`,
    `pandas`, `seaborn`
- Conventions for all cells
  - Widgets use short names: `mu`, `sigma`, `N`, `T`, `rho`, `gamma`, `q`, `seed`
  - `seed` is the last widget of each cell
  - Returns are in percent per period, characteristics are standardized
  - Each cell is one triplet: Goal and Description, widget code, Guided usage and
    Implementation

# Part 1: The Cross-Section and the Model

## Cell 1.1: Cross-Section Versus Time-Series Variation

**Goal**
- Separate the two sources of variation in returns: differences across securities and
  changes over time
  - Both procedures in this notebook explain the first one, not the second
- See that a characteristic can explain the cross-section of average returns even
  when the time series of returns is noisy
**Plots and their descriptions**
- `Cross-section`: one dot per security, x-axis is the time-averaged characteristic,
  y-axis is the time-averaged return, with a fitted line
- `Time series`: line of the cross-sectional mean return in each period, the common
  movement the procedures do not try to explain
- `Variance split`: stacked bar of return variance into the common time effect, the
  characteristic-driven spread across securities, and idiosyncratic noise
- `Comments`: current `mu`, `sigma`, `N`, `seed`, fitted slope, $R^2$ of the
  `Cross-section` scatter, and the three variance shares
**Widgets**
- `mu`: slider for the true premium per unit of characteristic, 0.0-1.0 percent per
  period
  - Tilts the `Cross-section` line and grows the characteristic share of variance
- `sigma`: slider for the idiosyncratic volatility, 1-10 percent per period
  - Spreads the dots around the line and grows the noise share
- `N`: logarithmic slider for the number of securities, 32-1024
  - Changes how many dots appear and how stable the fitted slope is
- `seed`: integer input, 0-999
  - Redraws the whole simulated universe
**Guided usage**
- Drag `mu` from 0.0 to 1.0, leaving `sigma` fixed
  - Observe the `Cross-section` scatter tilt into an upward line and the
    characteristic share in `Variance split` grow
  - Observe `Time series` stay unchanged
- Raise `sigma` from 2 to 10
  - Observe the dots spread around the line and $R^2$ fall, although the true `mu` is
    the same
- Raise `N` from 32 to 1024 with a few different `seed` values
  - Observe the fitted slope settle near `mu` instead of jumping between seeds
**Implementation**
- `cell1_1_cross_section_vs_time_series()` in the utils
- `pandas` panel of securities by periods, grouped with `groupby` for the averages
- `seaborn.regplot` and `seaborn.lineplot` for the panels
- `htutori.build_widget_control`, `htutori.build_log_widget_control`, and
  `htutori.add_fitted_text_box` for the controls and the `Comments` panel

## Cell 1.2: Anatomy of Returns, Factors, and Controls

**Goal**
- Name the pieces of $Y = X \beta + Z \gamma + \varepsilon$ on actual data: returns
  $Y$, explanatory factors $X$, and control variables $Z$
- Build on Cell 1.1: the cross-section there is now split into its columns
**Plots and their descriptions**
- `Y`: heat strip of the returns of $N$ securities in period `t`, one row per
  security
- `X`: heatmap of the $K$ characteristics of the same securities, one column per
  factor
- `Z`: heatmap of the control variables of the same securities
- `Comments`: current `t`, `K`, `N`, the true $\mu_k$ and $\gamma$, and the
  correlation between each column of `X` and `Y`
**Widgets**
- `t`: slider for the period shown, 1-T
  - Redraws `Y`, and moves `X` and `Z` only slowly
- `K`: slider for the number of factors, 1-3
  - Adds or removes columns in `X`
- `N`: slider for the number of securities shown, 8-64
  - Kept small so each cell of the heatmaps stays readable
- `seed`: integer input, 0-999
**Guided usage**
- Step `t` from 1 to 10
  - Observe `Y` change at every step while `X` and `Z` drift only slowly: the
    characteristics are persistent, the returns are not
- Raise `K` from 1 to 3
  - Observe new columns appear in `X` and the correlation list in `Comments` grow to
    one line per factor
- Sort the rows by the first column of `X` using the same `seed`
  - Observe `Y` darken from one end to the other: the pattern that the regression in
    Part 2 will measure
**Implementation**
- `cell1_2_returns_factors_controls()` in the utils
- `seaborn.heatmap` for the three matrices, with a shared diverging colormap
- `pandas.DataFrame.sort_values` for the sorting toggle
- `ipywidgets.Checkbox` for the sort toggle, sliders from `htutori`

# Part 2: Procedure 1, Cross-Sectional Regressions

## Cell 2.1: One-Period Cross-Sectional Regression

**Goal**
- Estimate the factor exposure $\hat\beta_t$ in one period by regressing the returns
  of all securities on their characteristics
- See that one period gives a noisy estimate, which motivates averaging in Cell 2.2
**Plots and their descriptions**
- `Scatter`: returns $Y$ against the first characteristic in period `t`, with the
  least-squares line, the true line with slope $\lambda_t$, and a dashed reference
  line with slope $\mu$
- `Estimates`: bar of $\hat\beta_{k,t}$ with a 95 percent interval for each factor,
  next to the true $\lambda_{k,t}$
- `Residuals`: histogram of the fitted residuals $\hat\varepsilon_i$
- `Comments`: current `t`, `N`, `sigma`, $\hat\beta$, its standard error, $R^2$, and
  the true $\lambda_t$
**Widgets**
- `t`: slider for the period, 1-T
  - Picks which cross-section is regressed
- `N`: logarithmic slider for the number of securities, 32-1024
  - Controls how many points enter the regression
- `sigma`: slider for the idiosyncratic volatility, 1-10 percent
- `seed`: integer input, 0-999
**Guided usage**
- Raise `N` from 32 to 1024 at a fixed `t`
  - Observe the interval in `Estimates` narrow and the fitted line approach the true
    line with slope $\lambda_t$
- Step `t` through ten consecutive periods at a fixed `N`
  - Observe $\hat\beta_t$ jump around the dashed $\mu$ reference, sometimes below
    zero: a single period is unreliable
- Raise `sigma` from 2 to 10
  - Observe the interval widen and $R^2$ fall, while the center of the line stays
    near the true line
**Implementation**
- `cell2_1_one_period_regression()` in the utils
- `statsmodels.api.OLS` with `add_constant`, reading `params`, `bse`, and
  `conf_int()`
- `seaborn.scatterplot`, `seaborn.histplot`, and `matplotlib` for the error bars
- Sliders from `htutori`

## Cell 2.2: Averaging Betas Across Periods

**Goal**
- Follow the Fama-MacBeth procedure end to end: one regression per period, then the
  mean and standard deviation of $\hat\beta_t$ across all periods
- Turn the mean and standard deviation into a $t$-statistic for the factor premium,
  building on the single regression of Cell 2.1
**Plots and their descriptions**
- `Beta path`: line of $\hat\beta_t$ over the periods, with a solid line at the
  sample mean and a dotted line at the true $\mu$
- `Histogram`: distribution of $\hat\beta_t$ over the periods, with the mean and the
  standard deviation marked
- `Running mean`: running mean of $\hat\beta_t$ with a band of plus or minus two
  standard errors, as the periods accumulate
- `Comments`: current `T`, `mu`, `sigma`, mean and standard deviation of
  $\hat\beta_t$, standard error $s / \sqrt{T}$, and the $t$-statistic
**Widgets**
- `T`: logarithmic slider for the number of periods, 16-512
  - Sets how many cross-sectional regressions are averaged
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `sigma`: slider for the idiosyncratic volatility, 1-10 percent per period
- `seed`: integer input, 0-999
**Guided usage**
- Set `mu` to 0.2 and raise `T` from 16 to 512
  - Observe the `Running mean` band narrow and the $t$-statistic grow roughly with
    $\sqrt{T}$
- Set `mu` to 0.0 and step through `seed` values 0 to 20
  - Observe the $t$-statistic usually stay inside plus or minus 2, but cross it for
    some seeds: a premium that does not exist can look significant
- Raise `sigma` from 2 to 10 at a fixed `T`
  - Observe `Beta path` get noisier and the $t$-statistic fall, while the mean stays
    near `mu`
**Implementation**
- `cell2_2_fama_macbeth()` in the utils
- Loop over periods calling `statsmodels.api.OLS`, collected into a `pandas.Series`
  of $\hat\beta_t$
- `pandas.Series.mean`, `std`, and `expanding` for the summary and the running mean
- `seaborn.lineplot` and `seaborn.histplot` for the panels

## Cell 2.3: Controls Remove Omitted-Variable Bias

**Goal**
- Show why the regression includes control variables $Z$: leaving out a control that
  is correlated with the factor biases $\hat\beta$
- Build on Cell 2.2 by running the same procedure twice, with and without $Z$
**Plots and their descriptions**
- `X versus Z`: scatter of the first characteristic against the control in one
  period, showing the correlation set by `rho`
- `Estimates`: grouped bars of the mean $\hat\beta$ without controls and with
  controls, each with a 95 percent interval, next to the true $\mu$
- `Beta path`: lines of $\hat\beta_t$ for both specifications over the periods
- `Comments`: current `rho`, `gamma`, `mu`, the two mean estimates, the bias of each,
  and the two $t$-statistics
**Widgets**
- `rho`: slider for the correlation between the factor and the control, 0.0-0.9
  - Sets how much of the control effect leaks into $\hat\beta$ when it is omitted
- `gamma`: slider for the effect of the control on returns, 0.0-1.0 percent per
  period
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `seed`: integer input, 0-999
**Guided usage**
- Set `gamma` to 0.5 and raise `rho` from 0.0 to 0.8
  - Observe the no-control bar drift away from the true $\mu$ while the with-control
    bar stays on target
  - Observe the with-control interval widen: a correlated control costs precision
- Set `gamma` to 0.0 with `rho` at 0.8
  - Observe both bars agree again: omitting a control matters only if the control
    drives returns
- Set `rho` to 0.0 with `gamma` at 0.5
  - Observe both bars agree again: an uncorrelated control does not bias the factor
**Implementation**
- `cell2_3_controls_and_bias()` in the utils
- `statsmodels.api.OLS` fitted twice per period, on `X` alone and on `X` plus `Z`
- `pandas.DataFrame` of both $\hat\beta_t$ series, averaged with `mean`
- `seaborn.barplot` with error bars and `seaborn.scatterplot`

# Part 3: Procedure 2, Sorted Portfolios

## Cell 3.1: Ranking Securities Into Quantile Portfolios

**Goal**
- Rank securities by a characteristic, partition the universe into quantile subsets,
  and compute the time series of average returns of each subset
- See the value idea directly: top-ranked securities outperform bottom-ranked ones
  when the premium is real
**Plots and their descriptions**
- `Ranking`: securities in one period sorted by the lagged characteristic, as colored
  bands, one color per quantile
- `Average return`: bar of the mean return of each quantile portfolio with an error
  bar, from lowest to highest quantile
- `Cumulative return`: lines of the cumulative return of each quantile portfolio over
  the periods
- `Comments`: current `q`, `mu`, `sigma`, securities per quantile, mean return of the
  top and the bottom quantile, and their spread
**Widgets**
- `q`: slider for the number of quantiles, 2-10
  - Sets how finely the universe is partitioned
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `sigma`: slider for the idiosyncratic volatility, 1-10 percent per period
- `seed`: integer input, 0-999
**Guided usage**
- Raise `mu` from 0.0 to 0.5 with `q` at 5
  - Observe `Average return` bars step up from the lowest to the highest quantile
  - Observe the `Cumulative return` lines fan out in the same order
- Set `mu` to 0.0
  - Observe the bars become flat within their error bars and the lines overlap
- Raise `q` from 3 to 10 at a fixed `mu`
  - Observe the top-minus-bottom spread widen but every error bar widen too: fewer
    securities in each portfolio
**Implementation**
- `cell3_1_quantile_portfolios()` in the utils
- `pandas.qcut` on the characteristic of the previous period, then `groupby` on
  period and quantile for the average returns
- `seaborn.barplot`, `seaborn.lineplot`, and `seaborn.scatterplot`
- Sliders from `htutori`

## Cell 3.2: Building the Long-Short Portfolio

**Goal**
- Turn the ranking of Cell 3.1 into one return series that isolates the factor:
  positive weights on top-ranked securities, negative weights on bottom-ranked ones
- See how the choice of legs and weights changes the factor return series
**Plots and their descriptions**
- `Weights`: bar of the portfolio weight against security rank, positive for the top
  leg, negative for the bottom leg, zero in between
- `Long-short return`: bars of the long-short return in each period, with the
  cumulative return as a line
- `Distribution`: histogram of the long-short returns, with the mean marked
- `Comments`: current `frac`, `scheme`, `mu`, sum of the weights, mean and standard
  deviation of the long-short return, and its $t$-statistic
**Widgets**
- `frac`: slider for the share of securities in each leg, 0.1-0.5
  - Widens or narrows both legs together
- `scheme`: dropdown with equal weight or rank weight
  - Switches how weights are spread within a leg
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `seed`: integer input, 0-999
**Guided usage**
- Raise `frac` from 0.1 to 0.5
  - Observe `Weights` fill the whole range with no zero-weight middle
  - Observe the mean long-short return fall, since the legs now include less extreme
    securities
  - Observe the standard deviation fall, since each leg holds more securities
- Switch `scheme` from equal weight to rank weight at `frac` of 0.3
  - Observe `Weights` change from flat steps to a ramp that favors the extremes
- Set `mu` to 0.0
  - Observe the `Distribution` center on zero and the cumulative line wander without
    a trend
**Implementation**
- `cell3_2_long_short_portfolio()` in the utils
- `pandas.Series.rank` for the ranks, with weights scaled so the long leg sums to +1
  and the short leg to -1
- `pandas.DataFrame.mul` and `groupby` on period for the portfolio return
- `ipywidgets.Dropdown` for `scheme`, `seaborn.barplot`, `seaborn.histplot`

## Cell 3.3: Time-Series Regression on Factor Returns

**Goal**
- Regress the return series of each quantile portfolio $Y$ on the long-short series
  $X$, plus a control series $Z$
- Show that the estimated $\beta$ differs across portfolios and lines up with their
  average returns: the factor explains differences in average returns
**Plots and their descriptions**
- `Scatter`: return of the selected quantile portfolio against the long-short return,
  one dot per period, with the least-squares line
- `Beta by quantile`: bar of $\hat\beta$ for each quantile portfolio with a 95
  percent interval
- `Mean return versus beta`: one dot per quantile portfolio, x-axis is $\hat\beta$,
  y-axis is the average return, with a fitted line
- `Comments`: current `quantile`, `control`, `mu`, $\hat\beta$, its $t$-statistic,
  the intercept, and $R^2$
**Widgets**
- `quantile`: dropdown for the portfolio shown in `Scatter`, lowest to highest
  - Moves the highlighted bar and dot in the other panels
- `control`: dropdown with none or market return as $Z$
  - Adds the market return series to the regression
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `seed`: integer input, 0-999
**Guided usage**
- Step `quantile` from the lowest to the highest portfolio
  - Observe the `Scatter` slope turn from negative to positive
  - Observe the highlighted point in `Mean return versus beta` climb the line
- Switch `control` from none to market return
  - Observe $\hat\beta$ stay put but its interval narrow and $R^2$ rise, since the
    market removes common noise
- Set `mu` to 0.0
  - Observe the lowest and highest portfolios keep a nonzero $\hat\beta$ because they
    are the legs of the factor, while the middle portfolios go to zero
**Implementation**
- `cell3_3_time_series_regression()` in the utils
- `statsmodels.api.OLS` fitted once per quantile portfolio, reading `params`,
  `tvalues`, `conf_int()`, and `rsquared`
- `seaborn.regplot` and `seaborn.barplot` for the panels
- `ipywidgets.Dropdown` for `quantile` and `control`

## Cell 3.4: Multifactor Sorts Partition Each Subset

**Goal**
- Extend the sort to two factors: partition by the first factor, then partition each
  subset again by the second factor
- See what a second factor changes in the portfolio returns, and when the two sorts
  interact, building on Cell 3.1
**Plots and their descriptions**
- `Mean return grid`: heatmap of the mean return in each cell of the `q` by `q`
  double sort, first factor on rows, second factor on columns
- `Count grid`: heatmap of the average number of securities in each cell
- `Marginal means`: two lines, the mean return by first-factor quantile and by
  second-factor quantile, each averaged over the other factor
- `Comments`: current `q`, `sort`, `rho`, `mu1`, `mu2`, the smallest and the largest
  cell count, and the mean-return range along each axis
**Widgets**
- `q`: slider for the number of quantiles per factor, 2-5
- `sort`: dropdown with sequential or independent sort
  - Sequential partitions each first-factor subset again, independent ranks both
    factors on the full universe
- `rho`: slider for the correlation between the two characteristics, 0.0-0.9
- `mu1`: slider for the premium of the first factor (value), 0.0-1.0 percent per
  period
- `mu2`: slider for the premium of the second factor (size), 0.0-1.0 percent per
  period
- `seed`: integer input, 0-999
**Guided usage**
- Set `mu1` to 0.5 and `mu2` to 0.0
  - Observe `Mean return grid` vary down the rows only, and the second-factor line in
    `Marginal means` go flat
- Set `sort` to independent and raise `rho` from 0.0 to 0.9
  - Observe `Count grid` collapse onto the diagonal, with near-empty off-diagonal
    cells
- Switch `sort` to sequential at `rho` of 0.9
  - Observe `Count grid` equalize again, as every subset is partitioned in full
**Implementation**
- `cell3_4_multifactor_sort()` in the utils
- `pandas.qcut` applied to the whole universe for the independent sort, and inside
  `groupby` for the sequential sort
- `pandas.DataFrame.pivot_table` for the two grids
- `seaborn.heatmap` and `seaborn.lineplot`, with `ipywidgets.Dropdown` for `sort`

# Part 4: Significance and Comparison

## Cell 4.1: Statistical Significance and False Discoveries

**Goal**
- Understand what "$\hat\beta$ is statistically significant" means for a researcher
  stating a case: a $t$-statistic beyond plus or minus 2
- See how often a factor with no premium still passes the test, and how the chance of
  detecting a real premium depends on `T`, building on Cell 2.2
**Plots and their descriptions**
- `t-stat distribution`: histogram of the Fama-MacBeth $t$-statistic over many
  simulated studies, with vertical lines at plus and minus 2 and the rejection zones
  shaded
- `Rejection rate`: curve of the share of studies with $|t| > 2$ against the true
  `mu`, with a marker at the current `mu`
- `Comments`: current `mu`, `T`, `n_trials`, share of studies with $|t| > 2$, and
  mean $t$-statistic
**Widgets**
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `T`: logarithmic slider for the number of periods per study, 16-512
- `n_trials`: logarithmic slider for the number of simulated studies, 32-1024
  - Controls how smooth the histogram and the rate estimate are
- `seed`: integer input, 0-999
**Guided usage**
- Set `mu` to 0.0 and raise `n_trials` to 1024
  - Observe about 5 percent of studies fall in the rejection zones although no
    premium exists
- Raise `mu` from 0.0 to 0.2 at `T` of 64
  - Observe the histogram shift right and the marker climb the `Rejection rate` curve
- Raise `T` from 16 to 256 at `mu` of 0.1
  - Observe the `Rejection rate` curve steepen: a small premium becomes detectable
    only with a long sample
**Implementation**
- `cell4_1_significance_and_false_discoveries()` in the utils
- Vectorized `numpy` slope and standard error per period, so many studies run without
  a loop over `statsmodels` fits
- `pandas.Series.mean` and `std` for the Fama-MacBeth $t$-statistic of each study
- `seaborn.histplot` and `seaborn.lineplot`, `htutori.build_log_widget_control`

## Cell 4.2: Comparing the Two Procedures

**Goal**
- Run both procedures on the same simulated studies and compare the premium estimates
  and the verdicts
- Close the arc: both procedures target differences in average returns across
  securities and reach similar conclusions by different routes
**Plots and their descriptions**
- `Estimates`: points with 95 percent intervals for the Fama-MacBeth mean $\hat\beta$
  and the rescaled mean long-short return, next to the true `mu`
- `t-stat scatter`: one dot per simulated study, x-axis is the Fama-MacBeth
  $t$-statistic, y-axis is the long-short $t$-statistic, with the 45 degree line and
  the plus or minus 2 thresholds
- `Verdicts`: bar of the share of studies significant under both, only Fama-MacBeth,
  only long-short, and neither
- `Comments`: current `mu`, `T`, `N`, `frac`, the correlation between the two
  $t$-statistics, and the share of studies where the verdicts disagree
**Widgets**
- `mu`: slider for the true premium, 0.0-1.0 percent per period
- `T`: logarithmic slider for the number of periods, 16-512
- `N`: logarithmic slider for the number of securities, 32-1024
- `frac`: slider for the share of securities in each long-short leg, 0.1-0.5
- `seed`: integer input, 0-999
**Guided usage**
- Raise `mu` from 0.0 to 0.3 at `T` of 64
  - Observe the cloud in `t-stat scatter` move up and right, with most studies ending
    significant under both procedures
  - Observe the two $t$-statistics stay strongly correlated across studies
- Lower `T` to 24 at a small `mu` of 0.1
  - Observe the `Verdicts` bar for disagreement grow: some studies pass under one
    procedure only
- Lower `frac` from 0.5 to 0.1
  - Observe the long-short $t$-statistic spread more, since the legs use fewer
    securities
**Implementation**
- `cell4_2_compare_procedures()` in the utils
- Reuses the vectorized Fama-MacBeth routine of Cell 4.1 and the long-short routine
  of Cell 3.2
- Rescales the mean long-short return by the characteristic spread between the legs,
  so both estimates are in return per unit of characteristic
- `seaborn.scatterplot`, `seaborn.barplot`, and `pandas.crosstab` for the verdicts