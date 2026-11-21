# Mean–Variance Portfolio Optimization with Neural Networks

MSML610 · Fall 2026 · [Project specification](../../../../../research/ideas/draft.Mean_Variance_Optimization_with_NN.md) · [Issue #760](https://github.com/gpsaggese/gpsaggese.github.io/issues/760)

Can a neural network allocate a portfolio faster than a convex solver? Does training a return predictor on portfolio utility improve investment decisions? This project studies both questions using nine U.S. sector ETFs. In the recorded experiment, the solver was faster in both constraint settings, and equal weighting delivered the highest out-of-sample net utility.

[Open the notebook](mvo_utils.ipynb) · [Read the experiment report](docs/experiment_report.md) · [Download the per-seed metrics](docs/strategy_metrics.csv)

## Reproduce the experiment

Use a checkout of the full course repository with Docker running. Start at the repository root and run the following commands. The Jupyter launcher uses the shared scripts in the `helpers_root` submodule.

```bash
git submodule update --init --recursive helpers_root
cd class_project/msml610/Fall2026/projects/UmdTask760_MSML610_Fall2026_Mean_Variance_Optimization_with_NN_1
DOCKER_ENGINE=docker ./docker_build.sh
DOCKER_ENGINE=docker ./docker_jupyter.sh
```

Open the URL printed by the launcher (default port 8888). To use port 8889, launch with `DOCKER_ENGINE=docker ./docker_jupyter.sh -p 8889`. Open `mvo_utils.ipynb` and select **Kernel → Restart Kernel and Run All Cells**. The first code cell installs the pinned experiment packages. A fresh checkout needs internet access for installation and the initial Yahoo Finance download. Subsequent runs reuse the adjusted prices and daily returns cached in `data/` after checking their dates and hashes.

## What the notebook does

| Stage | Method | Comparison |
| --- | --- | --- |
| 1. Statistical baselines | Estimate annualized mean returns and a Ledoit–Wolf covariance matrix from the previous 126 trading days. Solve long-only, fully invested portfolios with CLARABEL. | Equal weight, minimum variance, classical MVO, and MVO with historical mean shrinkage on the same dates after fees. |
| 2. Learn the solver’s weights | Train a network on 512 solver-labeled instances. Test on 128 instances from later dates, first with long-only weights, then with an additional L1 turnover cap of 0.4. | Compare the network and solver on each instance using the objective gap and median single-instance CPU time. |
| 3. Learn expected returns | Give both predictors each ETF’s 21-day mean return and 126-day volatility. Train one on return-prediction error and the other on realized gross portfolio utility through a differentiable MVO layer. | Convert both predictions to weights with the same MVO solver. Compare their net returns and utility with the Stage 1 baselines. |

The universe is XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, and XLY. The dataset contains 4,024 daily returns from January 4, 2010 through December 31, 2025. At each rebalance the portfolio is held for 21 trading days. Positions drift between rebalances, and both purchases and sales incur 10 basis points per dollar traded.

Stage 3 uses rolling folds with 60 holding periods for training, 12 for validation, and up to 12 for testing. The three seeds (7, 11, 19) share the same test dates. Stages 1 and 3 are evaluated from July 7, 2016 through December 11, 2025, covering 113 complete holding periods.

## Results

**Solver approximation.** A smaller objective gap means the network’s weights come closer to the solver’s optimum. The turnover cap narrowed the mean gap. CLARABEL had lower median latency in both settings on the benchmark CPU.

| Constraints | Mean objective gap | Network (ms per instance) | CLARABEL (ms per instance) |
| --- | ---: | ---: | ---: |
| Long-only, fully invested | 0.05518 | 0.738 | 0.530 |
| Same constraints with L1 turnover ≤ 0.4 | 0.02371 | 0.747 | 0.576 |

**Out-of-sample performance.** Equal weighting produced the highest compound annual growth rate (12.89%) and annual net utility (−0.0128). Minimum variance had the lowest annual volatility (14.50%). Across the three seeds, decision-focused training beat predict-then-optimize in two runs, but had lower average net utility (−0.0583 versus −0.0526). Annual net utility is annualized mean net return minus five times annualized net return variance.

**Historical mean shrinkage.** In notebook section 1.4, `classical_mvo` uses the full estimated mean return for each ETF and `shrunk_mean_mvo` uses 25% of that estimate. Shrinkage raised annual net utility from −0.1289 to −0.0632 (a change of +0.0658). Both strategies use the same Ledoit–Wolf covariance, constraints, test dates, and trading rules.

The seeds measure training variability over one market history. The results cover these nine ETFs and this test period.

## Outputs

The notebook writes per-rebalance weights, daily net returns, and per-seed backtests to `results/us_sector_etfs/`, along with the two Stage 2 models. The saved snapshot includes the data hashes, fold dates, and numerical results. [Experiment snapshot](docs/experiment_snapshot.json) · [Net portfolio value and drawdown for seed 7](docs/backtest_seed7.png)

## References

1. Zhang, Z., Zohren, S., & Roberts, S. (2020). [Deep Learning for Portfolio Optimization](https://arxiv.org/abs/2005.13665).
2. Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, J. Z. (2019). [Differentiable Convex Optimization Layers](https://proceedings.neurips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html).
3. Lee, J., Jeon, H., Bae, H., & Lee, Y. (2025). [Return Prediction for Mean-Variance Portfolio Selection: How Decision-Focused Learning Shapes Forecasting Models](https://doi.org/10.1145/3768292.3770423).
