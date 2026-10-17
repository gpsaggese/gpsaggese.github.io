# Milestone 3 — Data, Walk-Forward Folds & Baselines

> **GitHub Milestone**
> **Title:** `M3 — Data, Walk-Forward Folds & Baselines`
> **Timeframe:** Week 3 (Oct 12 – Oct 18)
> **Depends on:** `M2`
> **Description:**
> Build the numerical backbone and complete **spec Milestone 1** (container + data + baselines). Acquire clean, aligned daily returns for the nine-ETF basket since 2006 and cache them with a provenance note; define the **walk-forward folds** (5 training years + 1 test year, rolled yearly) as a non-leaking generator; and implement the two baselines — **equal-weight** and **closed-form minimum-variance** — reporting each test year's out-of-sample Sharpe and turnover (§6.1–§6.2). Every downstream estimate (`μ`, `Σ`, the optimization, baseline weights) is fed **only** its fold's training slice. Exit criterion: a per-test-year baseline OOS Sharpe/turnover table, populated from non-leaking folds.

## Notes
- **Basket is fixed:** SPY, EFA, EEM, TLT, IEF, LQD, GLD, VNQ, DBC — a deliberately cross-asset basket with interesting covariance (§4.8).
- **Window:** daily data from **2006 onward**, long enough for many 5y-train/1y-test folds (~13 test years through the mid-2020s); annualize by a 252-day factor (§4.8).
- **Pass `auto_adjust=True` explicitly and use `Close`** — in `yfinance==1.7.0` the OHLC are already adjusted and there is **no** `Adj Close` column; don't assume one exists (§6.1, §11 risks).
- **Handle the MultiIndex columns** — `yf.download()` returns MultiIndex columns by default (price-type then ticker), even for a single ticker; select the `Close` level for a (dates × tickers) frame, or pass `multi_level_index=False` (§6.1).
- **Treat the cached CSV as the source of truth.** `yfinance` serves *adjusted* history that can be restated later (splits, dividends, backfills), so commit a provenance note (fetch date + checksum) and have notebooks load from the cache, not re-fetch (§4.7).
- **Static fallback (spec-required):** the Ken French **"10 Industry Portfolios (Daily)"** library if Yahoo blocks requests. It provides daily *returns* for ten industry portfolios (not the nine ETFs), so the loader should consume returns directly and skip the price→return step; cache it with the same provenance discipline (§4.8).
- **Walk-forward is non-negotiable and non-leaking:** 5 training years, 1 test year, rolled one year at a time; training strictly precedes test; no test-year data touches any estimate (§6.1).
- **Risk-free rate for Sharpe = the training window's average T-bill rate, not 0** — short rates moved from ~0% to ~5% across 2006–2024, which changes Sharpe rankings. State it once and reuse it everywhere Sharpe appears (§6.2).

---

## Issues

### `M3-1` — Data acquisition & caching
- **Assignee:** @jake
- **Labels:** `data`, `yfinance`, `core`
- **Blocked by:** `M2-2`
- **Blocks:** `M3-2`

**Context.** Download adjusted prices (`auto_adjust=True`) for the nine-ETF basket from 2006, flatten the MultiIndex columns to a (dates × tickers) `Close` frame, align on common trading dates, handle missing values, compute daily returns, and cache to CSV with a provenance note so results are reproducible. Wire and document the Ken French returns fallback (§6.1, §4.8).

**Deliverables**
- `yf.download(tickers, start="2006-01-01", end=..., auto_adjust=True)`; the `Close` level extracted from the MultiIndex columns into a (dates × tickers) table, aligned on common dates, NaNs handled.
- Daily simple returns computed from the aligned prices; ETFs with history shorter than 2006 handled (shorten the common window to the latest inception, or document the trimmed start).
- Cached returns table in the mounted `data/` directory with a provenance note (fetch date + checksum) and a brief data-quality note.
- A loader that reads the cache rather than re-fetching; the Ken French 10-Industry **returns** fallback path implemented and documented (consumes returns directly, skips price→return).

**Definition of Done**
- [ ] Prices come from `auto_adjust=True` (no reliance on an `Adj Close` column); MultiIndex reduced to a clean (dates × tickers) `Close` frame.
- [ ] No missing values; all assets share one date index; return ranges are sane.
- [ ] Cached table + provenance note (fetch date + checksum) recorded (the cache itself stays git/docker-ignored).
- [ ] Notebooks/scripts load from the cache, not the network; the Ken French fallback is wired and documented.

---

### `M3-2` — Walk-forward fold construction
- **Assignee:** @jake
- **Labels:** `data`, `walk-forward`, `validation`, `core`
- **Blocked by:** `M3-1`
- **Blocks:** `M3-3`, `M4-1`

**Context.** Turn the returns table into the walk-forward folds every later milestone consumes: for each fold, 5 consecutive training years and the following 1 test year, rolled forward by one year. Expose them as a generator so each downstream estimate is fed only its fold's training slice — the single mechanism that makes the out-of-sample test honest (§6.1).

**Deliverables**
- A fold iterator/generator yielding `(train_returns, test_returns)` per fold: 5 training years, the next 1 test year, rolled yearly.
- A guarantee that training strictly precedes test, with no overlap or leakage; folds contiguous and rolled one year at a time.
- Fold boundaries surfaced (dates per fold) for the data-quality note.

**Definition of Done**
- [ ] Folds are contiguous, non-leaking (training strictly precedes test), and rolled yearly.
- [ ] The generator yields the expected number of folds for a 2006→ window (~13 test years).
- [ ] Each fold's training slice is the only input exposed to estimation code.

---

### `M3-3` — Baselines: equal-weight & closed-form minimum-variance
- **Assignee:** @jake
- **Labels:** `baselines`, `finance`, `scipy`, `core`
- **Blocked by:** `M3-2`
- **Blocks:** `M5-1`

**Context.** Establish the naive comparators the Nevergrad portfolios must beat. For each fold, compute the equal-weight (`1/N`) and closed-form minimum-variance (`w ∝ Σ⁻¹ 1`, renormalized; long-only via `scipy.optimize` if the unconstrained solution goes short) portfolios, apply each to the test year, and record out-of-sample Sharpe and turnover — the comparison backbone reused in M5 (§6.2).

**Deliverables**
- Per-fold equal-weight and closed-form minimum-variance weights (long-only, sum to 1).
- A per-test-year table of out-of-sample Sharpe and turnover for both baselines.
- The training-window average T-bill risk-free rate chosen and recorded in one place for reuse (§6.2); a shared turnover definition (`Σ|w_new − w_old|`) used everywhere.
- Baseline and portfolio-stats helpers added to `nevergrad_utils.py` (no duplicated math).

**Definition of Done**
- [ ] Baseline weights are valid (long-only, sum to 1) for every fold.
- [ ] Per-test-year OOS Sharpe + turnover table populated for both baselines (completes spec Milestone 1).
- [ ] Risk-free rate recorded and referenced from one place; turnover defined once and reused.
- [ ] Min-variance long-only handling (clip-renormalize or `scipy.optimize`) is documented.
