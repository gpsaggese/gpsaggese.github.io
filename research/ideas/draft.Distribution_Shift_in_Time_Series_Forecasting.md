# Empirical Analysis of Foundation Model Robustness to Distribution Shifts in Time-Series Forecasting

## Status
**Status:** draft  
**Complete Specs:** 0%  
**Assignee:** TBD

# Template B: Full Research Project

## Description
- **Core Problem**: Foundation models (LLMs, vision transformers, time-series
  models like Chronos, TimesFM) are trained on diverse datasets but often fail
  silently when deployed on data with different statistical properties,
  seasonality patterns, or value ranges
- **Key Angle**: Systematic measurement of how various types of distribution
  shifts (covariate shift, label shift, concept drift, temporal shift) affect
  forecasting accuracy across model families
- **Novelty**: Most benchmarks test on held-out test sets from the same
  distribution; this project measures _transfer_ robustness explicitly
- **Contribution**: Taxonomy of shift types, quantitative degradation curves,
  and practical detection methods for practitioners

## Project Objective
Real-world time-series data often violates the assumption that training and test
distributions are identical. Stock prices shift regimes, sensor networks degrade
over time, and seasonal patterns change. This project asks: **How robust are
state-of-the-art foundation models to different types of distribution shifts,
and can we predict or detect when a model will fail?**

We will:

1. Curate a suite of time-series datasets with documented distributional shifts
2. Benchmark foundation models (and classical baselines) on in-distribution vs
   shifted test splits
3. Quantify accuracy degradation per shift type
4. Develop early-warning metrics to flag when a deployed model is drifting

This addresses a critical gap for practitioners deploying forecasting models in
production

## Core Thesis
- **Conventional view**: A model trained on historical data will generalize to
  future data if test data comes from the same distribution
- **Empirical motivation**: Real-world time series experience regime shifts,
  concept drift, and seasonal changes that violate IID assumptions; most
  production failures are silent (model runs but predictions degrade)
- **Hypothesis**: Different shift types (covariate vs. label vs. concept drift)
  degrade model performance predictably, and this degradation can be detected in
  real time via statistical tests on prediction residuals
- **Goal**: A quantitative framework mapping shift type → expected performance
  drop, plus a dashboard tool for monitoring deployed models

## Dataset Suggestions
1. **UCI Time Series Archive: Multiple Shift Datasets**
   - Source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
   - Contains: 128 labeled time-series datasets (ECG, stock prices, sensor data,
     weather) with documented structural breaks and regime changes
   - Access: Free download
   - Why: Provides ground-truth shift points; enables controlled evaluation

2. **Kaggle: Energy Consumption Data (Electricity Load)**
   - Source:
     https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
   - Contains: Hourly electricity demand (kWh) from 2004–2018; clear seasonal,
     weekly, and annual cycles with structural breaks during crises
   - Access: Free with Kaggle account
   - Why: Real-world covariate shift from grid modernization, weather patterns,
     and policy changes

3. **Yahoo! Finance API: Stock Price Time Series**
   - Source: yfinance Python library (free, no key required)
   - Contains: OHLCV (open, high, low, close, volume) data for equities, crypto,
     indices with market regime shifts
   - Access: Free tier
   - Why: Extreme volatility shifts, sentiment-driven regime changes, and
     concept drift from macro events

4. **NOAA Climate Data Online**
   - Source: https://www.ncei.noaa.gov/cdo-web/
   - Contains: Daily temperature, precipitation, wind speed with long-term
     climate drift and seasonal anomalies
   - Access: Free registration; REST API available
   - Why: Explicitly documents climate shift; natural benchmarks for concept
     drift

5. **M4 Forecasting Competition Dataset**
   - Source: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
   - Contains: 100k time series (hourly, daily, weekly, monthly, yearly) with
     known train/test splits and documented anomalies
   - Access: Free download
   - Why: Standardized benchmarks; enables direct comparison to published
     baselines

## Tasks
1. **Formalize Distribution Shift Taxonomy**: Define and operationalize
   covariate shift, label shift, concept drift, and temporal shift in
   time-series context; create synthetic versions of each (e.g., scale shifts,
   seasonal reversals) to enable controlled evaluation

2. **Implement Shift Detection Baselines**: Code statistical tests for
   distribution shift (Kolmogorov-Smirnov, Wasserstein distance on rolling
   windows, ADWIN, DDM) and integrate into a monitoring framework

3. **Benchmark Foundation Models & Classical Baselines**: Train at least 3
   foundation models (Chronos, TimesFM, or fine-tuned LLMs) and 3 classical
   methods (ARIMA, Prophet, ESN) on in-distribution training data

4. **Evaluate on Shifted Test Splits**: For each dataset and each shift type,
   measure accuracy (MAE, RMSE, MAPE) on in-distribution vs. shifted test
   splits; record degradation curves

5. **Analyze Shift Sensitivity Profiles**: Compute rank correlations between
   model architectures' robustness to shifts; identify which model families are
   robust to which shift types

6. **Develop Real-Time Monitoring Dashboard**: Implement a tool that flags when
   deployed models are drifting; test threshold tuning to balance false
   positives vs. detection latency

## Expected Findings
1. **Concept drift causes the largest accuracy degradation**: Models trained on
   pre-2008 financial data will be ~30–50% less accurate on post-2008 data;
   seasonal concept drift is often undetected
2. **Foundation models generalize better to covariate shifts but not concept
   drift**: Pre-trained transformers show 10–20% smaller accuracy drop on
   scaling shifts but fail silently on seasonal reversals
3. **Classical ARIMA and Prophet are more interpretable but similarly brittle**:
   Both are robust to magnitude shifts but fail on regime changes; Prophet's
   built-in changepoint detection is moderately effective (~70% recall at 1%
   false positive rate)
4. **Residual-based tests detect drift weeks before accuracy degrades**:
   Wasserstein distance on prediction residuals flags model decay 2–4 weeks
   before MAE visibly increases

## Bonus Ideas
- **Adaptive Retraining**: Compare fixed retraining schedules vs
  drift-triggered retraining; quantify cost/benefit tradeoff
- **Transfer Learning for Shifts**: Pre-train models on synthetic
  shift-augmented data; measure zero-shot transfer to real shifts
- **Domain Randomization in Forecasting**: Do models trained on mixtures of
  shift types generalize better? Compare to standard data augmentation

## Extensions
- **Multi-Step-Ahead Forecasting**: How does shift sensitivity grow with
  forecast horizon? (e.g., 1-step vs. 30-step ahead)
- **Anomaly Detection as Early Warning**: Can reconstruction error from
  autoencoders predict model failure better than statistical tests?
- **Causal Analysis**: Which causal variables drive shifts? Can interventions on
  slow-moving variables predict regime changes?

## Policy / Practical Implications
- **Monitoring Requirements**: Production forecasting pipelines should implement
  drift detection; current best practices lag by 5+ years
- **Model Selection**: Organizations should choose models based on documented
  robustness to their expected shift types, not just accuracy on holdout data
- **Retraining Strategy**: Reactive retraining on new data is inefficient;
  proactive retraining triggered by detected shifts can reduce alert latency by
  50%+

## Useful Resources
- [Chronos: Pretrained Language Models for Forecasting](https://arxiv.org/abs/2401.16588)
  — Foundation model for time series; arXiv preprint
- [Concept Drift Adaptation in Time-Series Forecasting](https://ieeexplore.ieee.org/document/8215634)
  — Gama et al. (2018) IEEE TKDE
- [The UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
  — 128 labeled datasets with ground-truth anomalies
- [ADWIN: Adaptive Windowing for Change Detection](https://dl.acm.org/doi/10.1145/1642194.1642271)
  — Bifet & Gavalda (2007)
