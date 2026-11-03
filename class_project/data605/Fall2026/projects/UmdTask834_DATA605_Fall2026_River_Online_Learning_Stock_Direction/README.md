# River: Online Prediction of Next-Day Stock Direction

- Course: DATA605, Fall 2026
- Author: Brandon Wilson (@bzwilson)
- Tool: [River](https://riverml.xyz/), a Python library for online machine
  learning on data streams
- Project description:
  [`River_Project_Description.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/project_descriptions/DATA605/River_Project_Description.md)

## Summary
- Builds an online model that predicts whether a stock closes higher tomorrow,
  updating once per trading day as new prices arrive
- Replays 2010 to 2026 daily data for a basket of large caps exactly as it
  would have been seen in real time: predict at today's close, learn when
  tomorrow's close reveals the label
- Measures any predictive edge against baselines with statistical tests that
  account for correlated errors, and tracks how performance changes across
  market regimes

## Quick Start
- From the root of the repository, go to the project directory:
  ```bash
  > cd class_project/data605/Fall2026/projects/<this project's branch name>
  ```
- Build the Docker image:
  ```bash
  > ./docker_build.sh
  ```
- Start Jupyter Lab and open `localhost:8888`:
  ```bash
  > ./docker_jupyter.sh
  ```
- Run the notebooks in order (see below). The first run of the example
  notebook downloads about 34k rows from Yahoo Finance and caches them in
  `data/`; later runs are offline
- Run the unit tests (seconds, no network):
  ```bash
  > ./docker_cmd.sh "python -m pytest test/test_river_utils.py"
  ```
- For details on the Docker scripts, see the
  [project template README](/class_project/project_template/docker_scripts.README.md)

## Files
- [`river.API.ipynb`](river.API.ipynb): tour of the River API on a built-in
  dataset (online contract, pipelines, metrics, progressive validation, drift
  detection)
- [`river.example.ipynb`](river.example.ipynb): end-to-end stock-direction
  project
- [`river_utils.py`](river_utils.py): all reusable logic (data acquisition,
  cleaning, features, streaming, evaluation, statistical tests); notebooks
  only call these functions
- `river.API.py`, `river.example.py`: Jupytext-paired versions of the
  notebooks for readable diffs
- [`data/README.md`](data/README.md): data source and caching
- `test/test_river_utils.py`: fast unit tests on synthetic data, including
  leakage checks
- `test/test_docker_all.py`: runs both notebooks inside Docker
- `Dockerfile`, `requirements.txt`, `docker_*.sh`: container setup from the
  class project template

## Design Decisions

### Why online learning for this problem
- Markets are non-stationary: relationships that held in 2015 may not hold in
  2022. A batch model must be periodically retrained on a chosen window; an
  online model updates continuously and can be paired with drift detection
- This project tests whether that adaptivity actually helps, by comparing
  River models with walk-forward batch models on the same timeline

### Problem framing
- Target: `1` if the log return from close `t` to close `t+1` is positive
- Optional dead zone removes days with tiny moves, which are mostly noise
- Universe: SPY plus 7 large caps from different sectors; SPY also provides
  market-wide features
- Data: daily bars from Yahoo Finance; the other suggested sources are
  discussed in [`data/README.md`](data/README.md)

### Avoiding look-ahead bias
- Every feature for day `t` uses only prices up to the close of `t`; a unit
  test checks that deleting all data after `t` leaves the features up to `t`
  unchanged
- Features are standardized online by `preprocessing.StandardScaler` inside
  the River pipeline, never with full-sample statistics
- Labels are delayed by one trading day: on day `t` the model predicts all
  tickers, then learns only the labels of day `t-1`. A naive row-by-row loop
  over a multi-ticker panel would learn ticker A's day-`t` label before
  predicting ticker B on day `t`, which uses tomorrow's information. A unit
  test checks the delay
- Sanity check: on a synthetic random walk the pipeline scores about 50%;
  on synthetic data with built-in momentum it finds the signal

### Evaluation
- Prequential (test-then-train) evaluation replaces k-fold cross-validation,
  which would train on the future and test on the past
- Metrics: accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC,
  log loss, and rolling accuracy over time
- Baseline accuracy is the majority-class rate, not 50%
- Significance: a binomial test against the baseline, a moving-block
  bootstrap confidence interval (resampling blocks of days, since errors on
  the same day are correlated), and McNemar's test to compare two models

## Roadmap
- [x] Project scaffold from the class template, Docker setup
- [x] Data acquisition with caching and provenance metadata
- [x] Validation, cleaning, leakage-safe features and labels
- [x] Prequential evaluation with one-day label delay, significance tests
- [x] Unit tests (leakage, delay, noise/signal sanity checks)
- [ ] Baselines: majority class, persistence, walk-forward batch models
- [ ] River model comparison: Hoeffding (adaptive) trees, adaptive random
      forest, ensembles
- [ ] Drift detection with ADWIN and adaptation across regimes
- [ ] Feature ablation (returns only vs. technical indicators vs. market
      context)
- [ ] Economic evaluation: long/flat strategy with transaction costs
- [ ] Final write-up and video

## Changelog
- 2026-11-01: Project scaffold, data pipeline, evaluation framework
