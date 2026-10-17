# Milestone 5 — Out-of-Sample Evaluation

> **GitHub Milestone**
> **Title:** `M5 — Out-of-Sample Evaluation`
> **Timeframe:** Week 6 (Nov 2 – Nov 8)
> **Depends on:** `M4`
> **Description:**
> Test the selected Pareto portfolios on held-out years against the baselines — the honest verdict on whether the Nevergrad portfolios are worth the complexity. In each fold, select the **single front point with the highest training Sharpe**, apply its weights and rebalancing frequency to the next (test) year, and report realized **cumulative net return, out-of-sample Sharpe, maximum drawdown, and turnover** alongside the equal-weight and minimum-variance baselines from M3. Plot the training Pareto fronts (NGOpt vs RandomSearch) and the cumulative net-return curves vs baselines (§6.5). In-sample Pareto-optimality does **not** guarantee out-of-sample performance; this is where that gap is measured. Exit criterion: a per-fold OOS metric table vs baselines plus the cumulative-return figure, computed on test-year data only.

## Notes
- **The selection rule is fixed and stated:** highest **training** Sharpe → applied to the next test year, so the choice is reproducible, not cherry-picked (§6.5).
- **Risk-free rate per M3-3** (training-window average T-bill), reused for every Sharpe (§6.2).
- **Max drawdown** = largest peak-to-trough decline of the cumulative net-return series (§6.5).
- **Report per-fold *and* aggregate** (mean per metric across folds), so the write-up can state clearly whether and when Nevergrad beats the baselines out of sample (§6.5).
- **Test-year data only** feeds the realized metrics — the walk-forward discipline from M3 carries through here (§6.1, §6.5).

---

## Issues

### `M5-1` — Front → portfolio selection & test-year application
- **Assignee:** @jake
- **Labels:** `out-of-sample`, `evaluation`, `core`
- **Blocked by:** `M4-2`, `M3-3`
- **Blocks:** `M5-2`, `M5-3`

**Context.** For each fold, pick the highest-training-Sharpe point from the training Pareto front and apply its weights and rebalancing frequency to the next test year, producing the realized return path the metrics are computed from (§6.5).

**Deliverables**
- A selection routine returning the highest-training-Sharpe front point per fold (weights + rebalancing frequency).
- Application of the selected portfolio to the fold's test year, yielding a realized net-return series.
- The selection rule documented once and reused.

**Definition of Done**
- [ ] Exactly one front point selected per fold by the stated training-Sharpe rule.
- [ ] Selected weights + frequency applied to test-year data only (no leakage).
- [ ] A realized net-return series produced per fold.

---

### `M5-2` — OOS metrics vs baselines
- **Assignee:** @jake
- **Labels:** `out-of-sample`, `evaluation`, `reporting`, `core`
- **Blocked by:** `M5-1`
- **Blocks:** `M5-3`, `M5-4`, `M6-1`

**Context.** Compute the realized out-of-sample metrics for the selected Nevergrad portfolio and both baselines, per test year and aggregated, so the results section has concrete numbers to interpret (§6.5).

**Deliverables**
- Per-fold OOS metrics — cumulative net return, out-of-sample Sharpe, maximum drawdown, turnover — for the Nevergrad portfolio and the equal-weight and minimum-variance baselines (reusing the M3 risk-free rate and turnover definition).
- An aggregate (mean per metric across folds) alongside the per-year table.
- Metric helpers added to `nevergrad_utils.py` (no duplicated math).

**Definition of Done**
- [ ] All four metrics computed on test-year data only, for the portfolio and both baselines.
- [ ] Per-fold and aggregate tables populated for every test year.
- [ ] The comparison against baselines is explicit; where Nevergrad wins/loses is stated.

---

### `M5-3` — Figures: training fronts & cumulative-return curves
- **Assignee:** @jake
- **Labels:** `out-of-sample`, `visualization`, `core`
- **Blocked by:** `M5-1`, `M5-2`
- **Blocks:** `M7-1`, `M7-2`

**Context.** Produce the two required figures: the training Pareto fronts (NGOpt vs RandomSearch) and the cumulative net-return curves of the selected portfolio vs the baselines, generated reproducibly from cached data (§6.5).

**Deliverables**
- A Pareto-front figure overlaying the NGOpt and RandomSearch training fronts.
- A cumulative net-return figure: the selected Nevergrad portfolio vs equal-weight and minimum-variance.
- Plotting helpers in `nevergrad_utils.py`; figures regenerate from the cache with fixed seeds.

**Definition of Done**
- [ ] Both figures generated reproducibly from cached data.
- [ ] The front figure shows NGOpt vs RandomSearch; the return figure shows portfolio vs both baselines.
- [ ] Figures are labeled/captioned and ready for the report and video.

---

### `M5-4` — Passive buy-and-hold benchmark `[stretch]`
- **Assignee:** @jake
- **Labels:** `benchmark`, `finance`, `stretch`
- **Blocked by:** `M5-2`
- **Blocks:** —

**Context.** Optional extra comparator: add a passive buy-and-hold benchmark (e.g. 100% SPY, or a static 60/40 equity/bond mix drawn from the basket) to the out-of-sample comparison, as a familiar reference point beyond the two naive baselines. Clearly gated; SPY is already an asset in the basket, so frame it as a single-asset reference, not a new data source (§6.5).

**Deliverables**
- A passive benchmark return path over the same test years, from the already-cached data.
- Its OOS metrics (cumulative net return, Sharpe, max drawdown) added to the comparison table/figure.
- A one-line note on why a passive reference is informative alongside equal-weight and min-variance.

**Definition of Done**
- [ ] Benchmark computed from cached basket data (no new fetch) over the same test years.
- [ ] Its metrics placed in the OOS comparison with the in-sample/estimation caveats stated.
- [ ] Clearly gated as stretch — core M5 does not depend on it.
