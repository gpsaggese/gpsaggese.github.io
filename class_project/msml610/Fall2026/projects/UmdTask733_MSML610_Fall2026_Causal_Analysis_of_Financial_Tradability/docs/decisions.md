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
- **Status**: proposed (step 1)

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
- **Status**: proposed
