
## November 11, 2026 — Experiment 003: Neural Price Forecasting

### Objective
Evaluate whether recurrent and convolutional temporal models improve
next-observation commodity log-return forecasts over price-only baselines.

### Models
LSTM, GRU, and Temporal Convolutional Network (TCN).

### Data
FRED WTI, Brent, and Henry Hub.
Full-history, commodity-specific chronological evaluation.

### Observed Results
| Commodity | Best model overall | Test RMSE | OOS R2 |
|---|---|---:|---:|
| WTI | Ridge | 0.031203 | 0.005299 |
| Brent | TCN | 0.036206 | 0.004651 |
| Henry Hub | LSTM | 0.109875 | 0.034039 |

### Findings
- Ridge remains strongest for WTI.
- TCN achieves the lowest observed Brent return RMSE.
- LSTM achieves the lowest observed Henry Hub return RMSE.
- Improvement over simple references is modest.
- Better return RMSE does not always imply better directional accuracy.

### Limitations
- Neural-model seed stability not yet established.
- No paired bootstrap significance test yet.
- Results from one historical holdout are exploratory.
- Additional rolling-origin validation is required.

### Next
1. Consolidate all 30 model-commodity evaluations.
2. Repeat neural experiments across multiple seeds.
3. Evaluate PatchTST.
4. Conduct rolling-origin comparisons and uncertainty analysis.
