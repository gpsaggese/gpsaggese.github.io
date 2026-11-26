# Results for Nine U.S. Sector ETFs

Data period: 2010-01-04 to 2025-12-31 (4024 trading days).
Out-of-sample period: 2016-07-07 to 2025-12-11 (2373 trading days).

## Stage 1: Traditional Portfolio Strategies

| Strategy | cagr | annual_volatility | sharpe_rf_zero | max_drawdown | annual_utility | annual_turnover | maximum_target_weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal_weight | 0.1289 | 0.1726 | 0.789 | -0.367 | -0.0128 | 0.4297 | 0.1111 |
| minimum_variance | 0.0773 | 0.145 | 0.5866 | -0.3021 | -0.02 | 2.8368 | 0.7295 |
| classical_mvo | 0.0471 | 0.1972 | 0.3323 | -0.3823 | -0.1289 | 11.0201 | 1.0 |
| shrunk_mean_mvo | 0.0529 | 0.1597 | 0.4028 | -0.3232 | -0.0632 | 6.8286 | 0.7761 |

## Stage 2: Approximation Accuracy and CPU Time

| Strategy | objective_gap_mean | nn_single_median_ms | solver_single_median_ms |
|---|---:|---:|---:|
| long_only | 0.05518 | 0.73833 | 0.53035 |
| turnover_0.4 | 0.02371 | 0.74708 | 0.57612 |

The mean objective gap measures how closely the network matches the solver. The two timing columns show the median CPU time per test instance in milliseconds.

## Stage 3: Average Results Across Three Random Seeds

| Strategy | cagr | annual_volatility | sharpe_rf_zero | max_drawdown | annual_utility | annual_turnover | maximum_target_weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| classical_mvo | 0.0471 | 0.1972 | 0.3323 | -0.3823 | -0.1289 | 11.0201 | 1.0 |
| decision_focused | 0.0642 | 0.1636 | 0.4634 | -0.3801 | -0.0583 | 11.865 | 1.0 |
| equal_weight | 0.1289 | 0.1726 | 0.789 | -0.367 | -0.0128 | 0.4297 | 0.1111 |
| minimum_variance | 0.0773 | 0.145 | 0.5866 | -0.3021 | -0.02 | 2.8368 | 0.7295 |
| predict_then_optimize | 0.0655 | 0.1607 | 0.4755 | -0.3579 | -0.0526 | 11.7943 | 1.0 |
| shrunk_mean_mvo | 0.0529 | 0.1597 | 0.4028 | -0.3232 | -0.0632 | 6.8286 | 0.7761 |

`equal_weight` had the highest average annual net utility.

- Decision-focused learning achieved higher annual net utility than `equal_weight` in 0 of 3 seeds.
- Decision-focused learning achieved higher annual net utility than `minimum_variance` in 0 of 3 seeds.
- Decision-focused learning achieved higher annual net utility than `classical_mvo` in 3 of 3 seeds.
- Decision-focused learning achieved higher annual net utility than `shrunk_mean_mvo` in 2 of 3 seeds.
- Decision-focused learning achieved higher annual net utility than `predict_then_optimize` in 2 of 3 seeds.

### Historical Mean Shrinkage

The shrinkage baseline scales historical mean returns by 25% and uses the same covariance, constraints, and trading rules as classical MVO.

The annual net utility difference between shrunk-mean MVO and classical MVO was +0.0658. Decision-focused learning differed from shrunk-mean MVO by +0.0049.

This comparison measures historical mean shrinkage within classical MVO. Ledoit–Wolf covariance shrinkage is common to both portfolios.

![Net portfolio value and drawdown](backtest_seed7.png)

Adjusted prices for the nine surviving ETFs approximate dividend reinvestment. The backtest estimates trading costs from the value traded. Decision-focused training uses gross portfolio utility over each holding period.

All three seeds use the same market history. Results depend on the ETFs selected and the market conditions during the sample period. The study uses no separate final holdout period.
