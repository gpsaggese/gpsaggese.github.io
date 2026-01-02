## Retail Sales Story & Model Design

### Narrative Insights from Data
- **Weekend demand spikes**: Average normalized sales jump from weekday levels (~320) to >430 on Saturdays and >460 on Sundays, signaling stronger leisure-driven shopping.
- **Promotion intensity drives orders**: Even a single promoted SKU in the window more than doubles mean sales, and higher counts (e.g., 300+ items) correlate with order-of-magnitude surges, emphasizing promo coverage as a covariate.
- **Store traffic matters**: Transactions correlate with sales (≈0.21), so the model tracks log-transformed traffic as a soft proxy for local demand shocks.
- **Holiday effect is nuanced**: National/regional holidays came out only slightly negative overall because closures offset festive demand; treating them as binary flags still helps capture localized dips/spikes.
- **Category skew**: A handful of families (GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY) dominate revenue, so early experiments focus on them plus the top 10 stores to keep sequences dense.

### Feature Engineering & Windows
- **Normalization**: Sales use `log1p` + z-score scaling so the loss is less sensitive to outliers (>100k). Promotions/transactions also use log scaling before z-score.
- **Seasonality encodings**: Weekly (`sin/cos` for day-of-week) and yearly (`sin/cos` for month) harmonics allow the RNN to learn cyclical behavior without manual one-hot vectors.
- **Event flags**: `is_holiday` (national/regional) and `is_weekend` capture closures or demand bursts; promotions carry their own signal via counts.
- **Entity embeddings**: Scaled store/family IDs (0-1) inform the model of which time series is in context, letting one shared model learn cross-entity patterns.
- **Sliding windows**: Each (store, family) series is sorted chronologically, then windowed into a `context_length=30` day encoder input followed by a `horizon=7` day decoder target. Windows ending before `2017-04-01` go to train; later windows form validation.

### Model Architectures
Both architectures live in `lstm_gru_forecasting.py` and share the same data loader, optimizer, and evaluation pipeline.

1. **LSTM Forecaster**
   - **Input**: `(batch, 30, feature_dim)` context (feature_dim ≈ 11).
   - **Recurrent core**: Single-layer LSTM implemented via `jax.lax.scan`; hidden size 64 balances capacity and speed for the restricted subset.
   - **Readout**: Final hidden state → linear layer with 7 outputs (next-week normalized sales). The model predicts all horizons jointly, forcing it to keep multi-step structure in hidden state.
   - **Why**: LSTM gates handle long-range dependencies such as promotion campaigns starting several weeks earlier and multi-week holiday build-ups.

2. **GRU Forecaster**
   - Same input/output interface but uses a GRU cell (fewer parameters, no cell state). Works as a lighter baseline when capacity or overfitting is a concern.
   - Useful for sanity-checking whether the extra LSTM memory materially improves accuracy on this dataset.

### Optimization Loop (shared)
- **Loss**: Mean squared error on normalized targets; MAE tracked for interpretability.
- **Optimizer**: `optax.adam` with lr `3e-3`, batch size 512, epoch count 6 (quick comparative run). All steps JIT-compiled for GPU/CPU efficiency.
- **Metrics**: Both normalized-space metrics and inverse-transformed MAE/MSE (original sales units) reported for practical sense-making.
- **Best-checkpoint tracking**: Validation MSE determines model checkpoint used for the final metric dump.

### Intended Next Steps
1. Extend entity coverage beyond five families once runtime permits (script currently ready—update `TrainingConfig`).
2. Add external regressors (e.g., oil prices) by merging their z-scored values before windowing.
3. Integrate rolling feature importance or SHAP-style analyses to keep the “story” grounded in model behavior.
4. Once JAX is installed, run `python lstm_gru_forecasting.py` to train both architectures and log comparison metrics.
