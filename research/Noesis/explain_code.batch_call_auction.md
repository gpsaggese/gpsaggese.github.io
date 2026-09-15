# Class Description

## `batch_call_auction.py`

- `Bid`
  - Buyer order: (tasks, min tier, max latency, min reliability, max price)
- `Ask`
  - Seller order: (tasks, tier, typical latency/reliability, min price)
- `Fill`
  - One matched buyer/seller trade at a tier's uniform clearing price
- `TierClearResult`
  - Outcome of clearing one tier (price, fills, unfilled quantities)
- `OrderBookStore`
  - Abstract pluggable storage backend for pending bids/asks
- `_InMemoryOrderBookStore`
  - Default in-memory list-based `OrderBookStore`
- `OrderBook`
  - Batch call-auction book
  - Queues orders and clears tiers per round

# Class Interface

## `batch_call_auction.py`

- `Bid(dataclass)`
  - `__init__(self, buyer_id: str, n_tasks: int, c_level_min: str, l_max: float, r_min: float, p_max: float) -> None`
    - Validates and stores one buy order
- `Ask(dataclass)`
  - `__init__(self, seller_id: str, n_tasks: int, c_level: str, l_typical: float, r_typical: float, p_min: float) -> None`
    - Validates and stores one sell order
- `Fill(dataclass)`
  - Plain data holder, no custom methods
- `TierClearResult(dataclass)`
  - Plain data holder, no custom methods
- `OrderBookStore(abc.ABC)`
  - `add_bid(self, bid: Bid) -> None`
    - Queue a bid
  - `add_ask(self, ask: Ask) -> None`
    - Queue an ask
  - `get_bids(self) -> List[Bid]`
    - Return pending bids in submission order
  - `get_asks(self) -> List[Ask]`
    - Return pending asks in submission order
  - `clear(self) -> None`
    - Abstract, drop every stored bid/ask
- `_InMemoryOrderBookStore(OrderBookStore)`
  - `__init__(self) -> None`
    - Init empty in-memory bid/ask lists
  - `add_bid(self, bid: Bid) -> None`
    - Append bid to internal list
  - `add_ask(self, ask: Ask) -> None`
    - Append ask to internal list
  - `get_bids(self) -> List[Bid]`
    - Return copy of stored bids
  - `get_asks(self) -> List[Ask]`
    - Return copy of stored asks
  - `clear(self) -> None`
    - Reset bid/ask lists to empty
- `OrderBook`
  - `__init__(self, *, store: Optional[OrderBookStore] = None) -> None`
    - Init book, defaulting to in-memory store
  - `submit_bid(self, bid: Bid) -> None`
    - Queue a buyer order for next `clear_round()`
  - `submit_ask(self, ask: Ask) -> None`
    - Queue a seller order for next `clear_round()`
  - `get_pending_bids(self) -> List[Bid]`
    - Return buy orders queued for the next round
  - `get_pending_asks(self) -> List[Ask]`
    - Return sell orders queued for the next round
  - `clear_round(self) -> Dict[str, TierClearResult]`
    - Clear every tier present in the book and empty it

# Function Interface

## `batch_call_auction.py`

- `_match_orders_in_tier(c_level: str, bids: List[Bid], asks: List[Ask]) -> TierClearResult`
  - Matches one tier's bids/asks and computes uniform clearing price
  - Consumes `Bid`/`Ask` and produces a `TierClearResult` containing `Fill`s

# Class Relationships

## Inheritance

- `OrderBookStore`
  - `_InMemoryOrderBookStore` from `batch_call_auction.py`

## Composition

- `OrderBook`
  - Holds one `OrderBookStore`

## Uses

- `OrderBook.clear_round()` calls `_match_orders_in_tier()` per capability tier and
  returns `Dict[str, TierClearResult]`
- `TierClearResult` aggregates `List[Fill]`