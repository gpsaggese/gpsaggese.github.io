# `PR_M16`: Economic Surplus of Noesis vs a Posted-price Baseline - Implementation Spec

- This document specifies the library code needed to measure the economic surplus
  created by `NoesisMarket`'s batch call auction compared to a posted-price router
  in the style of OpenRouter, on the same orders
- Scope, as stated in the plan: quantify the benefit of a double-sided market for
  serving tokens, as proof that Noesis is better and as a familiar story for people
  who do not know market theory
  - Theory already says a double-sided market allocates better than posted prices
  - This PR adds the numbers, not new theory
- Roadmap position: `PR_M16` is unscheduled and not part of any `v0.x` milestone
  - Depends on `PR_M11` (merged): seeded `(bids, asks)` generators in
    `market/scenarios.py`
  - Does not need `PR_M10`: it calls `OrderBook.clear_round()` directly and works on
    `Fill` objects, so `run_round()` can replace the call later
  - `PR_M17` (notebook, sweep, chart, and marketing summary) depends on it
- This is a specification only: no code in this document has been implemented

## Design Decisions

### Surplus Is Computed From Fills

- Decision: one pure function computes surplus from the orders and a flat list of
  `Fill`, regardless of the mechanism that produced the fills
- Mapping to existing fields in `market/batch_call_auction.py`:
  - Buyer willingness to pay (WTP): `Bid.p_max`
  - Seller cost: `Ask.p_min`
  - Fill: `Fill.price` and `Fill.n_tasks`
- Definitions, summed over fills:
  - **Consumer surplus**: `n_tasks * (p_max - price)`
  - **Producer surplus**: `n_tasks * (price - p_min)`
  - **Total surplus**: consumer plus producer surplus, equal to
    `n_tasks * (p_max - p_min)`
  - **Maximum surplus**: total surplus of the best matching per tier, i.e., walk bids
    from the highest `p_max` and asks from the lowest `p_min` while `p_max >= p_min`
  - **Deadweight loss**: maximum surplus minus total surplus
- `Fill` stores `buyer_id` and `seller_id`, not the limit prices, so the function
  looks up `p_max` and `p_min` by id
  - Justification: `PR_M11` guarantees participant ids unique within each side of a
    generated scenario, and the function asserts it for any other input
- Total surplus does not depend on the price: the price only splits the surplus
  between buyers and sellers
  - The gain therefore comes from which trades happen and who trades with whom

### Posted-price Baseline With the Same Output Type

- Decision: `run_posted_price_round(bids, asks, markup)` returns `List[Fill]`, so
  the surplus function is shared and the comparison is like for like
- Rules, per capability tier (`c_level`), matching the exact tier bucket that
  `OrderBook.clear_round()` uses:
  - Each seller posts the price `p_min * (1 + markup)`
  - Process bids in list order (the order is seeded by the scenario generator)
  - Route each bid to the cheapest ask with remaining capacity
  - Buy only if the posted price is at most `p_max`: otherwise the bid is dropped
  - Fill `min(remaining bid n_tasks, remaining ask n_tasks)`, then continue with the
    next cheapest ask until the bid or the eligible capacity is exhausted
  - Each fill pays its seller's posted price: no uniform price
- `markup` is a non-negative float
  - `0` is the best case for the baseline: seller posts at cost
  - TODO(gp): pick the default markup from real OpenRouter prices and check the
    actual OpenRouter routing rule (cheapest, fastest, fallback order)
- The function is named for the mechanism, not for OpenRouter: it models a posted
  price router, not OpenRouter's production behavior

### Scenario Generator With Overlapping Prices

- Decision: add `make_surplus_scenario` to `market/scenarios.py`, next to the
  `PR_M11` generators, with the same conventions (keyword-only arguments, local
  `random.Random(seed)`, fresh objects, unique participant ids, `hdbg.dassert_*`
  validation)
- Why a new generator: `PR_M11` families draw bids and asks from separated integer
  bands (e.g., bids 6 to 9, asks 1 to 4) to guarantee a regime, so every bid crosses
  every ask or none does; surplus differences between mechanisms need some pairs to
  cross and some not
- Signature:

  ```python
  def make_surplus_scenario(
      *,
      seed: int = 0,
      n_bids: int = 20,
      n_asks: int = 20,
      tiers: Sequence[str] = ("standard",),
      wtp_to_cost_ratio: float = 1.5,
      price_sigma: float = 0.3,
  ) -> Scenario: ...
  ```

- Draws per order: `n_tasks` uniform in 1 to 100; `p_max` lognormal with median
  `wtp_to_cost_ratio`; `p_min` lognormal with median 1; both with log standard
  deviation `price_sigma`
  - `wtp_to_cost_ratio` controls how much of the book crosses
  - Prices are continuous floats, so ties are not a concern
- All declared terms (`l_max`, `r_min`, `l_typical`, `r_typical`) use the compatible
  values of the `PR_M11` families: the auction matches on tier and price only

### Comparison Helper

- Decision: `compare_mechanisms(bids, asks, markup) -> pd.DataFrame`, one row per
  mechanism (`noesis`, `posted_price`) with consumer, producer, total surplus,
  maximum surplus, deadweight loss, and matched volume
- It builds an `OrderBook`, submits copies of the orders, calls `clear_round()`,
  flattens the `TierClearResult.fills`, and calls the baseline on the same lists
  - Justification: one entry point for the notebook in `PR_M17`, and one place that
    guarantees both mechanisms see the same book

## Trade-off and Alternative Design

- **Result type**: baseline returns `List[Fill]` rather than `TierClearResult`
  - A posted-price round has no clearing price, so `TierClearResult` would carry a
    meaningless field
  - Trade-off: the baseline cannot report unfilled quantity per tier, which the
    surplus metrics do not need
- **Closed-form surplus** (integrate the demand and supply curves) instead of
  simulation: not chosen, because the order books in `PR_M11` are discrete and the
  simulation reuses the real auction code
- **Distribution choice**: lognormal prices are an assumption, not data
  - Alternative: anchor `p_min` on real OpenRouter price lists, tracked as a
    `TODO(gp)` above and in `PR_M17`

## Out of Scope

- Notebook, parameter sweep, charts, and the marketing summary: `PR_M17`
- Strategic bidding: the surplus is computed on the submitted limits, which are
  assumed to be the true WTP and cost
  - A uniform-price auction is not strategy-proof, so the Noesis result is an upper
    bound under truthful bids
- Latency and reliability scoring: the auction matches on tier and price only, so the
  baseline does too
- Live providers, API calls, database, or network
- Changes to auction pricing or matching in `batch_call_auction.py`

## Current State

- `market/batch_call_auction.py`:
  - `Bid` (line 39) has `p_max`, `n_tasks`, `c_level_min`
  - `Ask` (line 94) has `p_min`, `n_tasks`, `c_level`
  - `Fill` (line 160) has `buyer_id`, `seller_id`, `c_level`, `n_tasks`, `price`
  - `_match_orders_in_tier` (line 218): sort bids descending and asks ascending,
    match while the best bid is at least the best ask, uniform price at the
    midpoint of the marginal bid and ask
  - `OrderBook.clear_round()` (line 435): buckets by tier, returns
    `Dict[str, TierClearResult]`
- `market/scenarios.py`: `make_balanced_scenario` (line 147) to
  `make_random_scenario` (line 349); returns `Scenario = Tuple[List[Bid], List[Ask]]`
- Gaps this spec closes:
  - No surplus metric exists
  - No posted-price mechanism exists to compare against
  - No generator draws overlapping bid and ask prices

## Implementation

- Two new files in `research/Noesis/market/`: `surplus.py` and
  `posted_price_baseline.py`
- One function added to `scenarios.py`; no change to existing functions
- `market/` keeps importing no other Noesis component

### `surplus.py`: Surplus Metrics

- Interface:

  ```python
  @dataclasses.dataclass(frozen=True)
  class SurplusResult:
      consumer: float
      producer: float
      total: float
      maximum: float
      deadweight_loss: float
      matched_n_tasks: int

  def compute_surplus(
      bids: List[Bid], asks: List[Ask], fills: List[Fill]
  ) -> SurplusResult: ...

  def compare_mechanisms(
      bids: List[Bid], asks: List[Ask], markup: float
  ) -> pd.DataFrame: ...
  ```

- `compute_surplus` asserts unique ids per side, that every fill id exists, and that
  `p_min <= price <= p_max` for every fill (individual rationality)
- `maximum` is computed per tier with the greedy walk, so it needs no solver

### `posted_price_baseline.py`: Posted-price Router

- Interface:

  ```python
  def run_posted_price_round(
      bids: List[Bid], asks: List[Ask], markup: float = 0.0
  ) -> List[Fill]: ...
  ```

- Pure function: does not mutate the input orders and keeps remaining capacity in
  local state

## Interaction with Existing Code

- New call sites only: `compare_mechanisms` calls `OrderBook.submit_bid`,
  `submit_ask`, and `clear_round`
- No existing function changes behavior, so existing tests in `market/test/` must
  pass unchanged

## Configuration and Secrets

- Not applicable

## Unit Test Plan

- Follow `.claude/skills/testing.rules.md`
- `market/test/test_surplus.py`
  - One bid, one ask, `p_max > p_min`: total is `n_tasks * (p_max - p_min)`,
    consumer plus producer equals total
  - No cross (`PR_M11` `make_no_cross_scenario`): every surplus is zero and
    deadweight loss is zero
  - Noesis result: `total == maximum` and `deadweight_loss == 0` over 30 seeds
  - Every fill has `p_min <= price <= p_max`
  - Duplicate ids or an unknown id in a fill raise `AssertionError`
- `market/test/test_posted_price_baseline.py`
  - One bid, one ask, posted price at most `p_max`: same fill as Noesis, same total
  - Same case with posted price above `p_max`: no fill
  - Bid routed to the cheapest ask first, and to the next ask when capacity ends
  - `markup=0` total is at most Noesis total over 30 seeds
  - Total is non-increasing as `markup` grows, on a fixed seed
  - Input orders are not mutated
- `market/test/test_scenarios.py`: add cases for `make_surplus_scenario`
  - Same seed returns identical orders, different seeds differ
  - Global random state is unchanged
  - A book with `wtp_to_cost_ratio` well above 1 has crossing pairs, and one well
    below 1 has few
  - Invalid arguments raise `AssertionError`
- Existing `PR_M11` tests must keep passing unchanged

## Risks and Limitations to Call Out

- Truthful bidding is assumed: the Noesis surplus equals the maximum by
  construction when limits are true values
  - The headline number compares the mechanisms, not strategies
- The gain depends on `markup`, `wtp_to_cost_ratio`, and `price_sigma`
  - `PR_M17` must report a range from a sweep, never one number
- OpenRouter is a posted-price market, not a centrally planned one
  - The gain may be small when `markup` is near zero
  - Marketing text must not call OpenRouter centrally planned
- Lognormal draws are an assumption, so synthetic books may favor either mechanism
  - Mitigation: publish the generator and prefer real price data when available

## Result (to Fill in Once Implemented)

- Placeholder: record what was built, what was deferred, and any deviation from
  this spec
