# Market

- `NoesisMarket`: a batch call-auction that matches buyer bids and seller asks for
  LLM inference capacity, per capability tier, at one uniform price
- Cleared fills become `Contract`s, dispatched to a stubbed fulfillment layer
- `market/` imports no other Noesis component
- Code walkthrough: `docs/notes/explain_code.batch_call_auction.md`

## Structure of the Dir

| Dir     | Description                                  |
| :------ | :------------------------------------------- |
| `test/` | Unit tests, one `test_*.py` per module       |

## Description of Files

| File                  | Description                                                           | Cluster  |
| :-------------------- | :-------------------------------------------------------------------- | :------- |
| `batch_call_auction.py` | `Bid`, `Ask`, `OrderBook`, and per-tier uniform-price clearing      | Auction  |
| `scenarios.py` | Named and seeded bid/ask fixtures | Scenarios |
| `contract_dispatch.py` | `Contract` schema, `build_contracts()`, and stubbed fulfillment dispatch | Dispatch |
| `supply_adapter.py` | Pure adapter: OpenRouter endpoint prices -> one `Ask` per seed seller, posted once per round | Supply |

## Description of Architecture

- `contract_dispatch.py` imports `batch_call_auction.py` (`Bid`, `Ask`,
  `TierClearResult`); not the reverse
- `OrderBook` keeps its orders behind the `OrderBookStore` interface (in-memory by
  default); the Postgres implementation lives in `noesis_platform/postgres_store.py`
- `supply_adapter.py` imports `batch_call_auction.py` (`Ask`, `OrderBook`) and takes
  the OpenRouter catalog as parsed JSON; the HTTP fetch and the seed file live in
  `noesis_platform/supply_feed.py` (`docs/decisions.md` D-25 to D-27)

## Deterministic Scenario Families (PR_M11)

Named generators vary quantities and prices by seed while guaranteeing their
advertised auction regime. The five types are families of examples, not five
fixed books. All return fresh `(bids, asks)` lists without external I/O.

| Function | Guaranteed regime |
| :------- | :---------------- |
| `make_balanced_scenario()` | Equal compatible supply and demand; all fill |
| `make_excess_demand_scenario()` | All supply fills; buyer quantity remains |
| `make_excess_supply_scenario()` | All demand fills; seller quantity remains |
| `make_multi_tier_scenario()` | All tiers fill at distinct prices |
| `make_no_cross_scenario()` | All bid prices below ask prices; no fills |
| `make_random_scenario(seed, n_bids, n_asks, tiers)` | No guaranteed regime |

### Usage

```python
import research.Noesis.market.batch_call_auction as rnbacaau
import research.Noesis.market.scenarios as rnmascen

bids, asks = rnmascen.make_excess_demand_scenario(
    seed=42, n_bids=3, n_asks=2, quantity_scale=1000, price_scale=100
)
book = rnbacaau.OrderBook()
for bid in bids:
    book.submit_bid(bid)
for ask in asks:
    book.submit_ask(ask)
results = book.clear_round()

multi_bids, multi_asks = rnmascen.make_multi_tier_scenario(
    seed=7, n_bids=2, n_asks=3, tiers=("standard", "frontier", "premium")
)
```

### Parameters and Construction

- All optional named-family arguments are keyword-only, including `seed`
- Named families default to seed 0 and two orders on each side per tier. Identical
  arguments reproduce values; change the seed to explore a different book.
- `n_bids` and `n_asks` are positive integer counts per tier. Single-tier families
  accept `tier`; the multi-tier family accepts at least two distinct tier names.
- Default examples stay small; larger counts are allowed for demonstrations.
- Quantities are drawn from 1 to 100, then the final order on the smaller side
  is increased to equalize totals. This adjusted quantity may exceed 100.
- Excess-demand/supply families then add a random positive quantity to one order
  on the appropriate side. Multiplying by `quantity_scale` preserves the regime.
- Crossing books draw asks from 1–4 and bids from 6–9, so every bid can afford
  every ask. No-cross reverses these bands. Multi-tier offsets each tier's band
  by 20, guaranteeing distinct clearing prices. `price_scale` multiplies prices.
- Prices are illustrative integer-valued floats. Scales are positive integers;
  they do not establish behavior for arbitrary floating-point extremes.
- Buyers use a 2-second limit and 90% reliability; sellers use 1-second typical
  latency and 95% reliability. These are compatible declared terms, not measures.
- Seeds are integers; counts and scales exclude booleans. Tier names must be
  nonblank strings. Multi-tier names must be ordered and unique. Invalid inputs
  raise `AssertionError` through the repository's assertion helpers.
- Participant ids are unique within each side, and every call returns independent
  objects. A local random generator leaves global randomness unchanged.

The unrestricted `make_random_scenario` retains its existing required arguments.
It permits zero orders, draws quantity 1–100 and price 1–20 independently, and does
not balance totals or enforce crossing. It complements the regime-specific families.

### Validation and M10 Dependency

Tests exercise each named family across 30 seeds, three count/scale configurations,
including one-order books and large numeric scales. They check reproducibility,
fresh objects, quantity conservation, regime guarantees, distinct multi-tier prices,
price limits, and deterministic contract dispatch. The unrestricted generator also
has validation and invariant checks across 100 seeded books.

M10's `run_round()` remains unavailable in this checkout. Integration through it
is still pending; current tests use the auction and existing contract functions.
No matching, database, scheduling, or provider behavior is changed by this module.
