# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NoesisMarket: An In-Memory Auction Round
#
# - Explore all five named auction regimes with small, seeded orders
# - The pedagogical arc: input orders, clearing, tables, and supply/demand curves
# - Fulfillment is **MOCKED**: every contract records `True`; no model is called,
#   no payment occurs, and these results do not demonstrate real delivery

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import logging

# %%
import helpers.hnotebook as hnotebook
import market_in_memory_utils as utils

import research.Noesis.market.auction_report as rnmaaurp
import research.Noesis.market.scenarios as rnmascen

# Initialize the standard notebook configuration and loggers.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - M11 supplies orders; M10 clears a fresh `OrderBook` and stores contracts in
#   memory; M12 supplies the tables and curves
# - Bid prices are buyer limits; ask prices are seller limits; quantities count
#   tasks. Prices are illustrative units per task, latency is seconds, and
#   reliability is a fraction from 0 to 1
# - Each example starts with a fresh store, so round IDs restart at 0; all tiers
#   within that example share the same round ID
# - Change `seed` and rerun the following cells to explore different values
#   while retaining each named regime

# %%
# Keep one editable seed for reproducible comparisons across regimes.
seed = 7
print(f"seed={seed}")

# %% [markdown]
# # Part 1: Named Auction Scenarios

# %% [markdown]
# ## Cell 1.1: Balanced Supply and Demand
#
# - Every bid and ask fills; both unmatched totals are zero.

# %%
# Build the named inputs and display the submitted orders.
bids, asks = rnmascen.make_balanced_scenario(seed=seed)
print("orders=")
display(rnmaaurp.orders_to_df(bids, asks))

# %%
# Run one mocked round and inspect the clearing results.
result = utils.run_auction(bids, asks)
utils.display_result(bids, asks, result)

# %% [markdown]
# ## Cell 1.2: Excess Demand
#
# - All supply fills; some buyer quantity remains unmatched.

# %%
# Build the named inputs and display the submitted orders.
bids, asks = rnmascen.make_excess_demand_scenario(seed=seed)
print("orders=")
display(rnmaaurp.orders_to_df(bids, asks))

# %%
# Run one mocked round and inspect the clearing results.
result = utils.run_auction(bids, asks)
utils.display_result(bids, asks, result)

# %% [markdown]
# ## Cell 1.3: Excess Supply
#
# - All demand fills; some seller quantity remains unmatched.

# %%
# Build the named inputs and display the submitted orders.
bids, asks = rnmascen.make_excess_supply_scenario(seed=seed)
print("orders=")
display(rnmaaurp.orders_to_df(bids, asks))

# %%
# Run one mocked round and inspect the clearing results.
result = utils.run_auction(bids, asks)
utils.display_result(bids, asks, result)

# %% [markdown]
# ## Cell 1.4: Multiple Tiers
#
# - Both tiers fill at distinct prices within one round.

# %%
# Build the named inputs and display the submitted orders.
bids, asks = rnmascen.make_multi_tier_scenario(seed=seed)
print("orders=")
display(rnmaaurp.orders_to_df(bids, asks))

# %%
# Run one mocked round and inspect the clearing results.
result = utils.run_auction(bids, asks)
utils.display_result(bids, asks, result)

# %% [markdown]
# ## Cell 1.5: Non-Crossing Prices
#
# - No bid can afford an ask; volume is zero and no clearing price is marked.

# %%
# Build the named inputs and display the submitted orders.
bids, asks = rnmascen.make_no_cross_scenario(seed=seed)
print("orders=")
display(rnmaaurp.orders_to_df(bids, asks))

# %%
# Run one mocked round and inspect the clearing results.
result = utils.run_auction(bids, asks)
utils.display_result(bids, asks, result)

# %% [markdown]
# ## Summary: The Mental Model
#
# - Compare the submitted totals with matched plus unmatched volume on each side
# - A partially filled order appears in the unmatched table with only its residual
#   quantity; no-cross retains all submitted quantities and creates no contracts
# - Curves use the original orders, with one plot per tier; a horizontal line
#   marks the clearing price only when trades occur
# - Mocked fulfillment always succeeds here. This notebook does not test provider
#   performance, payments, persistence, or HTTP integration
