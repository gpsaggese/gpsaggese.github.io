# Intelligence Market

## Status
- **Status:**: draft
- **Complete Specs:**: 20%

## Core Idea
- Treat "intelligence" (LLM inference capacity at a given capability,
  latency, and reliability level) as a fungible commodity, similar to
  electricity, and build a marketplace where buyers (apps and agents that
  need inference) and sellers (model and compute providers) trade contracts
  for it
- The marketplace has two phases:
  1) an auction that matches demand and supply into contracts
  2) contract fulfillment, where infrastructure monitors whether the delivered
     intelligence and performance match what was sold (see
     [[draft.Intelligence_Server]] for the fulfillment and monitoring layer)
- The non-obvious part is that the traded unit is not raw compute (FLOPs or
  GPU-hours): it bundles a capability level with latency and reliability
  guarantees, so the market needs a way to quote and verify "how good", not
  just "how much"

## Formalization
- A contract is a tuple:
  ```text
  contract = (N_tasks, C_level, L_max, R_min, P)
  ```
  where
  - `N_tasks` is the number of tasks (tokens or task-equivalents)
  - `C_level` is the required capability tier
  - `L_max` is the maximum latency
  - `R_min` is the minimum reliability (fraction of tasks completed within
    `C_level` and `L_max`)
  - `P` is the price

- Every `T` minutes (default `T = 5`), run a call auction over the open
  order book:
  - A buyer submits a bid:
    `(N_tasks, C_level_min, L_max, R_min, P_max)`
  - A seller submits an ask:
    `(N_tasks, C_level, L_typical, R_typical, P_min)`
  - Bids and asks are bucketed by capability tier, then matched
    highest-bid-first against lowest-ask-first within a tier, clearing at a
    single uniform price per tier (as in a call auction), until supply and
    demand cross

- Notation:
  ```text
  clearing_price(C_level, t) = uniform price at auction round t for tier
                                C_level
  ```

## Key Examples
- **Basic auction round**: a buyer wants 10,000 tasks at "frontier-class"
  capability, under 2 seconds latency, 99.9% reliability; several sellers
  post asks at that tier; the auction clears at one uniform price for the
  tier and matches the buyer to one or more sellers
- **Tiered commodity**: three capability tiers (cheap, medium, frontier)
  clear independently each round, analogous to peak and off-peak pricing
  in electricity markets
- **Default**: a seller wins a contract, then fails to fulfill it (measured
  reliability drops below `R_min`); the fulfillment layer
  ([[draft.Intelligence_Server]]) must detect this and feed it back into
  future pricing or eligibility, the same way a capacity market penalizes
  non-delivery

## Questions
1. What is the right unit of "task" for cross-provider comparability:
   tokens, wall-clock compute, or a benchmark-normalized task-equivalent?
2. Is a uniform-price call auction the right mechanism at this frequency and
   scale, or would a continuous double auction or order book serve
   latency-sensitive buyers better?
3. How does the market prevent gaming: a seller misrepresenting its
   capability tier, or a buyer shading bids?
   - Without a slow onboarding or reputation system that defeats the point of an
     open market?
4. Should pricing be denominated in real currency or a synthetic
   "task-credit", and how much does that choice change adoption and
   regulatory surface?
5. Is a 5-minute batch auction too slow for latency-sensitive apps, and does
   a hybrid design (spot market plus batch auction) make sense, mirroring
   day-ahead versus real-time electricity markets?

## Research Topics
- Auction and market microstructure: call auctions, continuous double
  auctions, uniform versus discriminatory pricing, applied to a bundled
  capability/latency/reliability good instead of a single scalar
- Commodity and futures market design, adapted from electricity markets
  (day-ahead, real-time, and capacity markets) to intelligence contracts
- Reputation and slashing mechanisms for sellers that under-deliver, which
  require ground-truth performance measurement from
  [[draft.Intelligence_Server]]
- Mechanism design against gaming: bid shading, capability
  misrepresentation, and collusion between a small number of sellers
- Related market designs: an on-chain prediction market
  ([[draft.Create_Prediction_Market_on_blockchain]]) shares the
  matching-and-settlement problem, minus the capability-verification part

## Next steps
- [ ] Look for related research (what has already been done)
- [ ] Finalize the implementation plan
- [ ] GP to review / approve the plan
- [ ] Hack a quick end-to-end prototype (e.g., in 1-2 days) to show that you
      understood the problem and can make progress
- [ ] Break the problem down in phases and milestones
- [ ] Execute one step at the time

## Implementation plan
- Milestone 1: build a minimal batch call-auction simulator
  - In-memory order book, bid/ask submission, per-tier uniform-price
    clearing
  - This is the result: a unit-tested auction that clears correctly on
    synthetic order books, with no real settlement yet

- Milestone 2: define the contract schema and dispatch fulfillment
  - Define the `(N_tasks, C_level, L_max, R_min, P)` contract schema and
    hand off a cleared contract to [[draft.Intelligence_Server]] for
    execution
  - This is the result: a closed loop where a matched contract is
    dispatched and fulfillment is logged back to the market

- Milestone 3: add a reputation and pricing feedback loop
  - Feed measured performance (from the fulfillment layer) back into
    seller eligibility and pricing for future auction rounds
  - This is the result: sellers that under-deliver are priced out or
    excluded from subsequent rounds

## References
- Electricity market design: day-ahead and real-time double auctions with
  uniform clearing prices: TBD, needs a specific reference
- Related idea: [[draft.Create_Prediction_Market_on_blockchain]]
