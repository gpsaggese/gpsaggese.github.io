# Causal Analysis of Financial Tradability Decisions

## Data

### D-1. Use Binance's public market data endpoint

- **Decision**: download prices from `data-api.binance.vision`
- **Why**: `api.binance.com` returns HTTP 451. The public
  endpoint serves the same Binance data. Binance.US works too, but its volume
  is about 500x smaller (about 0.6 BTC vs 290 BTC per hour)
- **Alternatives**: Binance.US; Kaggle `mczielinski/bitcoin-historical-data`
  (used in #381, ends in 2021)
- **Status**: proposed (asked on #733)

### D-2. Fail loudly when data is missing

- **Decision**: raise an error if the download fails or returns no data
- **Why**: the code in #381 silently switched to synthetic data, so results
  could look real without being real
- **Alternatives**: fall back to cached or synthetic data
- **Status**: accepted

### D-3. Separate bar size from trading horizon

- **Decision**: use 5-minute bars as the base data and define horizons (5m,
  15m, 1h, 4h, 1d) as multiples of it
- **Why**: in #381 the data frequency and the trading horizon were the same
  thing, so the effect of the horizon could not be studied on its own
- **Alternatives**: one dataset per frequency, as in Exercise 1
- **Status**: accepted (step 1)

### D-7. Download 5-minute bars from the Binance monthly files

- **Decision**: use the monthly zip files on `data.binance.vision` and check
  each one against its published SHA-256 checksum
- **Why**: the API returns 1000 bars per request at about 1.4 s each, so
  5-minute data takes about 15 minutes per coin. The monthly files take about
  1 minute
- **Alternatives**: the API from D-1, which is still used for 1h and 1d
- **Status**: accepted

### D-8. Do not fill long gaps; split the data into segments

- **Decision**: fill gaps of up to 3 bars with the last close, leave longer
  gaps empty, and start a new segment after each one. Features and labels are
  computed inside a segment
- **Why**: filling an outage of several hours would create fake flat prices,
  and a return across the gap would mix hours of moves into one bar
- **Alternatives**: forward-fill everything; drop the gap rows (then a shift
  by n rows is no longer n bars)
- **Status**: accepted

### D-9. Flag extreme returns, do not drop them

- **Decision**: flag |r| > 10 times the past day's std, keep the bar
- **Why**: all the flagged bars checked are real market events, and fat tails
  are part of what we study (see D-6)
- **Alternatives**: drop or winsorize them
- **Status**: accepted

### D-10. Use Yahoo Finance for SPY

- **Decision**: SPY prices from Yahoo Finance, adjusted for dividends and
  splits
- **Why**: free, no API key. Daily data goes back to 1993
- **Alternatives**: paid data (Polygon, Alpaca) for longer hourly history
- **Status**: proposed. Yahoo only has hourly bars for the last 730 days

## Simulation

### D-4. A wrong call loses the move

- **Decision**: PnL = sum((2 * hit - 1) * |r|)
- **Why**: a plain dot product of 0/1 hits with returns treats a wrong call as
  zero instead of a loss, and matches the spec's E[r] = (2p - 1) g
- **Alternatives**: hit . r as written in Exercise 1
- **Status**: proposed (asked on #733)

### D-5. Report both ways of drawing hits

- **Decision**: simulate "shuffle" (exactly round(p * N) hits, Exercise 1) and
  "bernoulli" (each hit independent, spec formula)
- **Why**: they give different answers at 1d (about 0.52 vs 0.525 for a 95%
  chance of profit), and the spec formula assumes bernoulli
- **Alternatives**: only one of them
- **Status**: accepted

### D-6. Use an effective number of trades in the p* formula

- **Decision**: p* = 1/2 + z_q / (2 sqrt(N_eff)) with
  N_eff = (sum |r|)^2 / sum r^2
- **Why**: with N, p* is too optimistic because of fat tails (only about 80%
  chance of profit at the 1h p* instead of 95%). With N_eff it matches the
  simulated 95% crossing for BTC and ETH at 1h and 1d
- **Alternatives**: the spec formula with N
- **Status**: proposed. Also matches the simulation for SPY

### D-11. Evaluate on non-overlapping decision times

- **Decision**: for a horizon h, keep one decision time every h
- **Why**: with 5-minute bars, the 1h forward returns of consecutive bars
  share 11 of 12 bars, so they are not independent trades and N would be
  12x too large
- **Alternatives**: use every bar and correct the standard errors
- **Status**: accepted

## Models

### D-12. Walk forward by year, with purging

- **Decision**: for each test year from 2022, fit on all data before it and
  leave out the last horizon before the test year
- **Why**: a label at t uses prices up to t + h, so without the gap the last
  training labels would use prices from the test year (López de Prado, Ch. 7)
- **Alternatives**: one train/test split; purged k-fold cross-validation
- **Status**: accepted

### D-13. Assume a 10 bp cost per round trip

- **Decision**: subtract 10 bp from each trade's return
- **Why**: in the range of Binance spot fees for a round trip, and the value
  used in the related work listed on #733
- **Alternatives**: report the breakeven cost only (the PnL per trade before
  costs, which is also in the results)
- **Status**: proposed

### D-14. Judge models by the weighted hit rate and PnL

- **Decision**: report the share of |r| traded in the right direction next to
  the plain hit rate
- **Why**: the logistic regression beats p* on the plain hit rate but makes no
  money, because it is right on small moves and wrong on big ones
- **Alternatives**: plain hit rate only, as in the spec
- **Status**: proposed
