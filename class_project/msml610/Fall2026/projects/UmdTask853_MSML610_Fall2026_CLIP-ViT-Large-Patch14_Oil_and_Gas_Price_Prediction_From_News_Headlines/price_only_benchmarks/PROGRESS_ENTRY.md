## November 11, 2026 — Experiment 004: PatchTST-style Price-Only Transformer

### Objective
Test whether a compact patch-based Transformer improves next-observed-date
WTI/Brent/Henry Hub log-return forecasting beyond the existing 10-model
price-only leaderboard, on identical chronological test rows.

### Implementation
- New `price_only_benchmarks/patchtst_model.py` and `patchtst` option in
  `price_only_benchmarks/benchmark_models.py`.
- Architecture: 30 past observed returns -> sliding patches (length 6,
  stride 3) -> 2-layer 4-head Transformer encoder -> flatten regression head.
- Two past-only auxiliary features: momentum_5 and volatility_20.
- Hyperparameters selected on validation only; seed 42; train+validation
  refit after selection; no test-set fitting.
- Based on the key PatchTST concepts, not a reproduction of official PatchTST.

### Validation
- 6/6 local automated tests passed, including 3 new Transformer tests.
- End-to-end smoke test passed on three *synthetic* price CSVs.
- Real FRED RMSE not yet measured for this model; enter it after WSL training.

### Evidence / commands
- `PYTHONPATH=price_only_benchmarks python -m pytest -q --noconftest price_only_benchmarks/test/test_patchtst.py price_only_benchmarks/test/test_benchmark_models.py`
- `python price_only_benchmarks/benchmark_models.py --period full --models patchtst --output-dir results/price_only_patchtst`

### Next
- Train on the three real downloaded FRED CSVs, preserving logs and configs.
- Merge baseline leaderboard after verifying identical test rows.
- Report model RMSE, MAE, OOS R², directional accuracy; later repeat seeds
  and add walk-forward/backtesting before claiming robust superiority.
