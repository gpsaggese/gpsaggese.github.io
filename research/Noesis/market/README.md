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
| `contract_dispatch.py` | `Contract` schema, `build_contracts()`, and stubbed fulfillment dispatch | Dispatch |

## Description of Architecture

- `contract_dispatch.py` imports `batch_call_auction.py` (`Bid`, `Ask`,
  `TierClearResult`); not the reverse
- `OrderBook` keeps its orders behind the `OrderBookStore` interface (in-memory by
  default); the Postgres implementation lives in `noesis_platform/postgres_store.py`
