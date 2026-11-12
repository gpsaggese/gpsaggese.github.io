# Foundation Model Robustness to Distribution Shifts in Time-Series Forecasting

## Status

- **Status:**: draft
- **Complete Specs:**: 90%

## Core Idea

- Foundation models (time-series models like Chronos and TimesFM, fine-tuned
  LLMs) are trained on diverse datasets but often fail silently when deployed on
  data with different statistical properties, seasonality patterns, or value
  ranges: the model keeps running, but its predictions degrade
- Most benchmarks test on a held-out split from the same distribution, so they
  do not measure _transfer_ robustness; real-world series instead show regime
  shifts, sensor degradation, and seasonal changes that violate the IID
  assumption
- Hypothesis: different shift types (covariate shift, label shift, concept
  drift, temporal shift) degrade forecasting accuracy in predictable ways, and
  the degradation can be detected in real time with statistical tests on the
  prediction residuals
- Contribution: a taxonomy of shift types, quantitative degradation curves per
  model family, and practical drift-detection methods for practitioners
  deploying forecasting models in production

## Formalization

- Let $(x_t, y_t)$ be the (history window, target) pair at time $t$, drawn from
  $P_t(x, y)$, and let $P_{train}$ and $P_{test}$ be the train and test
  distributions
- Shift types, defined by what changes between $P_{train}$ and $P_{test}$:
  - **Covariate shift**: $P(x)$ changes, $P(y \mid x)$ is fixed (e.g., a scale
    shift of the input series)
  - **Label shift**: $P(y)$ changes, $P(x \mid y)$ is fixed
  - **Concept drift**: $P(y \mid x)$ changes (e.g., a regime change)
  - **Temporal shift**: $P_t$ depends on the time index through a seasonal or
    regime component (e.g., a seasonal reversal)
- Degradation of model $m$ under shift type $s$, for an error metric
  $\text{Err} \in \{\text{MAE}, \text{RMSE}, \text{MAPE}\}$:
  ```
  Delta(m, s) = Err_m(shifted_s) / Err_m(in_distribution) - 1
  ```
- Residual-based drift score: with residuals $e_t = y_t - \hat{y}_t$, a reference
  window $R$, and a recent window $W_t$ of length $w$
  ```
  D_t = Wasserstein(dist(e in R), dist(e in W_t)),  alarm if D_t > tau
  ```
  - Evaluate by detection latency (time between the true shift and the alarm)
    and false-positive rate as a function of $\tau$

## Key Examples

- **Electricity load**: hourly demand has clear seasonal, weekly, and annual
  cycles; grid modernization, weather patterns, and policy changes introduce
  covariate shift that a model trained on earlier years has not seen
- **Stock prices**: market regime changes and macro events change
  $P(y \mid x)$, so a model trained on one regime can degrade on the next (concept
  drift)
- **Synthetic shifts**: apply a controlled scale shift or a seasonal reversal
  to a clean series, so that the shift type and its onset are known exactly
- **Silent failure**: the model returns forecasts with no error, and accuracy
  drops only after the shift; a residual-based alarm is the only signal

## Questions

1. Does each shift type degrade accuracy in a predictable, model-family-specific
   way? A counterexample is a robustness ranking of the models that flips
   between datasets for the same shift type.
2. Do foundation models generalize better than classical baselines (ARIMA,
   Prophet) to covariate shift but not to concept drift, or are both families
   equally brittle?
3. Can residual-based tests (e.g., Wasserstein distance) flag a model decay
   before the error visibly increases, and with what lead time at an acceptable
   false-positive rate?
4. If true, what does this change? Organizations would select models by their
   documented robustness to the expected shift types, not by holdout accuracy,
   and would trigger retraining by detected drift, not by a fixed schedule.

## Research Topics

- **Shift taxonomy**: operationalize the four shift types for time series, and
  generate a synthetic version of each (e.g., scale shifts, seasonal reversals)
- **Shift detection baselines**: Kolmogorov-Smirnov test, Wasserstein distance
  on rolling windows, ADWIN, and DDM, integrated into a monitoring framework
- **Models**: at least 3 foundation models (Chronos, TimesFM, or fine-tuned
  LLMs) and 3 classical methods (ARIMA, Prophet, ESN)
- **Datasets**: candidate sources, to check for usable shift points:
  - UCR Time Series Archive: 128 labeled datasets (ECG, sensor, and others),
    https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - Kaggle hourly energy consumption (2004-2018), real covariate shift from
    grid modernization, weather, and policy changes
  - Yahoo! Finance via the `yfinance` library: OHLCV data for equities, crypto,
    and indices with market regime shifts
  - NOAA Climate Data Online: daily temperature, precipitation, and wind speed
    with long-term climate drift, https://www.ncei.noaa.gov/cdo-web/
  - M4 Forecasting Competition: 100k series (hourly to yearly) with known
    train/test splits, which allows comparison to published baselines
- **Adaptive retraining**: fixed retraining schedules vs drift-triggered
  retraining, and the cost/benefit trade-off
- **Robust training**: pre-train on synthetic shift-augmented data and measure
  zero-shot transfer to real shifts; compare domain randomization over mixed
  shift types to standard data augmentation
- **Forecast horizon**: how the shift sensitivity grows from 1-step to 30-step
  ahead forecasts
- **Early-warning alternatives**: autoencoder reconstruction error vs
  statistical tests as a predictor of model failure
- **Causal analysis**: which causal variables drive the shifts, and whether
  interventions on slow-moving variables predict regime changes

## Next steps

- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan

- Milestone 1: shift taxonomy and detection baselines
  - Define the four shift types and generate a synthetic version of each on
    clean series
  - Code the detection baselines (KS, rolling-window Wasserstein, ADWIN, DDM)
  - This is the result: a labeled set of shifted series and a detector library
    with known detection latency on synthetic shifts

- Milestone 2: benchmark on shifted test splits
  - Curate the datasets and train 3 foundation models and 3 classical methods
    on in-distribution data
  - For each dataset and shift type, measure MAE, RMSE, and MAPE on the
    in-distribution and shifted test splits
  - This is the result: degradation curves $\Delta(m, s)$ for every model and
    shift type

- Milestone 3: shift sensitivity profiles
  - Compute rank correlations between the robustness of the model families
  - Identify which model families are robust to which shift types
  - This is the result: a mapping from shift type to expected performance drop

- Milestone 4: real-time monitoring
  - Implement a dashboard that flags when a deployed model is drifting
  - Tune the alarm threshold to balance false positives against detection
    latency
  - This is the result: a monitoring tool and a measured lead time of the
    residual-based alarm over the visible accuracy drop

## References

- Ansari et al., _Chronos: Learning the Language of Time Series_. (2024)
- Gama et al., _A Survey on Concept Drift Adaptation_. (2014)
- Dau et al., _The UCR Time Series Archive_. (2019)
- Bifet and Gavalda, _Learning from Time-Changing Data with Adaptive Windowing_.
  (2007)
