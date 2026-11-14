# Causal Analysis of Financial Tradability Decisions

## Data

### `D1` (Use Binance's public market data endpoint)

- **Decision**: download prices from `data-api.binance.vision`
- **Why**: `api.binance.com` returns HTTP 451. The public
  endpoint serves the same Binance data. Binance.US works too, but its volume
  is about 500x smaller (about 0.6 BTC vs 290 BTC per hour)
- **Alternatives**: Binance.US; Kaggle `mczielinski/bitcoin-historical-data`
  (used in #381, ends in 2021)
- **Status**: accepted (GP in the research gdoc, 2026-11-07: Binance and
  Yahoo are fine)

### `D2` (Fail loudly when data is missing)

- **Decision**: raise an error if the download fails or returns no data
- **Why**: the code in #381 silently switched to synthetic data, so results
  could look real without being real
- **Alternatives**: fall back to cached or synthetic data
- **Status**: accepted

### `D3` (Separate bar size from trading horizon)

- **Decision**: use 5-minute bars as the base data and define horizons (5m,
  15m, 1h, 4h, 1d) as multiples of it
- **Why**: in #381 the data frequency and the trading horizon were the same
  thing, so the effect of the horizon could not be studied on its own
- **Alternatives**: one dataset per frequency, as in Exercise 1
- **Status**: accepted (step 1)

### `D7` (Download 5-minute bars from the Binance monthly files)

- **Decision**: use the monthly zip files on `data.binance.vision` and check
  each one against its published SHA-256 checksum
- **Why**: the API returns 1000 bars per request at about 1.4 s each, so
  5-minute data takes about 15 minutes per coin. The monthly files take about
  1 minute
- **Alternatives**: the API from `D1`, which is still used for 1h and 1d
- **Status**: accepted

### `D8` (Do not fill long gaps; split the data into segments)

- **Decision**: fill gaps of up to 3 bars with the last close, leave longer
  gaps empty, and start a new segment after each one. Features and labels are
  computed inside a segment
- **Why**: filling an outage of several hours would create fake flat prices,
  and a return across the gap would mix hours of moves into one bar
- **Alternatives**: forward-fill everything; drop the gap rows (then a shift
  by n rows is no longer n bars)
- **Status**: accepted

### `D9` (Flag extreme returns, do not drop them)

- **Decision**: flag |r| > 10 times the past day's std, keep the bar
- **Why**: all the flagged bars checked are real market events, and fat tails
  are part of what we study (see `D6`)
- **Alternatives**: drop or winsorize them
- **Status**: accepted

### `D10` (Use Yahoo Finance for SPY)

- **Decision**: SPY prices from Yahoo Finance, adjusted for dividends and
  splits
- **Why**: free, no API key. Daily data goes back to 1993
- **Alternatives**: paid data (Polygon, Alpaca) for longer hourly history
- **Status**: accepted (GP, 2026-11-07). Yahoo only has hourly bars for the
  last 730 days, so the SPY 1h file is saved in the project Drive folder (see
  `data_sources.reference.md`)

### `D15` (Treat zero-volume bars right before a gap as part of the gap)

- **Decision**: in `clean_bars()`, a run of zero-volume bars that ends right
  before a missing bar is marked `is_stale` and treated as missing
- **Why**: 41 of the 42 zero-volume bars come right before a Binance outage.
  Trading had already stopped, so their prices are stale
- **Alternatives**: only flag them (before this change)
- **Note**: whether a bar is stale depends on the next bars being missing. It
  only affects the bars around the 3 outages, not the features or the models
  elsewhere
- **Status**: accepted (GP, 2026-11-07). A separate gap analysis is in #904

### `D16` (Drop the first hourly bar of each day for SPY)

- **Decision**: for hourly SPY, drop the return of the first bar of each New
  York trading day
- **Why**: that bar goes from the previous close, so it includes the overnight
  move, and the overnight dynamics are very different from the intraday ones.
  It is 14% of the bars but 54% of the sum of r^2
- **Alternatives**: keep it; treat it as its own horizon; split the day into
  regimes (opening, mid-day, closing), maybe later
- **Status**: accepted (GP, 2026-11-07)

## Simulation

### `D4` (A wrong call loses the move)

- **Decision**: PnL = sum((2 * hit - 1) * |r|)
- **Why**: a plain dot product of 0/1 hits with returns treats a wrong call as
  zero instead of a loss, and matches the spec's E[r] = (2p - 1) g
- **Alternatives**: hit . r as written in Exercise 1
- **Status**: proposed (GP is double-checking, 2026-11-07)

### `D5` (Report both ways of drawing hits)

- **Decision**: simulate "shuffle" (exactly round(p * N) hits, Exercise 1) and
  "bernoulli" (each hit independent, spec formula)
- **Why**: they give different answers at 1d (about 0.52 vs 0.525 for a 95%
  chance of profit), and the spec formula assumes bernoulli
- **Alternatives**: only one of them
- **Status**: accepted

### `D6` (Use an effective number of trades in the p* formula)

- **Decision**: p* = 1/2 + z_q / (2 sqrt(N_eff)) with
  N_eff = (sum |r|)^2 / sum r^2
- **Why**: with N, p* is too optimistic because of fat tails (only about 80%
  chance of profit at the 1h p* instead of 95%). With N_eff it matches the
  simulated 95% crossing for BTC and ETH at 1h and 1d
- **Alternatives**: the spec formula with N
- **Status**: proposed. Also matches the simulation for SPY

### `D11` (Evaluate on non-overlapping decision times)

- **Decision**: for a horizon h, keep one decision time every h
- **Why**: with 5-minute bars, the 1h forward returns of consecutive bars
  share 11 of 12 bars, so they are not independent trades and N would be
  12x too large
- **Alternatives**: use every bar and correct the standard errors
- **Status**: accepted

## Models

### `D12` (Walk forward by year, with purging)

- **Decision**: for each test year from 2022, fit on all data before it and
  leave out the last horizon before the test year
- **Why**: a label at t uses prices up to t + h, so without the gap the last
  training labels would use prices from the test year (López de Prado, Ch. 7)
- **Alternatives**: one train/test split; purged k-fold cross-validation
- **Status**: accepted

### `D13` (Assume a 10 bp cost per round trip, and sweep it)

- **Decision**: subtract 10 bp from each trade's return by default, and also
  report the results for 0, 2, 5, 10, and 20 bp
- **Why**: in the range of Binance spot fees for a round trip, and the value
  used in the related work listed on #733. GP: 10 bp is on the expensive
  side, and the real cost depends on how aggressive the orders are, the
  quality of execution, and the time frame
- **Alternatives**: report the breakeven cost only (the PnL per trade before
  costs, which is also in the results)
- **Status**: accepted (GP, 2026-11-07)

### `D14` (Judge models by the weighted hit rate and PnL)

- **Decision**: report the share of |r| traded in the right direction next to
  the plain hit rate
- **Why**: the logistic regression beats p* on the plain hit rate but makes no
  money, because it is right on small moves and wrong on big ones
- **Alternatives**: plain hit rate only, as in the spec
- **Status**: proposed. Whether the results hold with a different target is
  checked in #905

## Scope

### `D17` (Describe what happens before asking why)

- **Decision**: for now, measure how tradability changes with the horizon,
  the cost, and the market conditions, without a causal analysis
- **Why**: GP (2026-11-07): focus on the "what" first, then investigate the
  "why" later
- **Alternatives**: causal analysis of the horizon effect by regime, with
  placebo tests (the original step 3)
- **Status**: accepted (GP, 2026-11-07)
