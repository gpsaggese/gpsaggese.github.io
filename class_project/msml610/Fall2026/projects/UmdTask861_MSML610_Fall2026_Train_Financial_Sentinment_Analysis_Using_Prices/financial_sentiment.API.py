# %% [markdown]
# # Financial sentiment using prices: reproducible API
# Author: @ypeng12 · MSML610 Fall2026 · Course issue #861.
#
# Two inputs are deliberately separate: observed numerical features support an
# actual baseline refit; saved public reports support an integrity audit of the
# 2023 BERT experiment and 2026 strategy comparison. Loading a report does not
# rerun BERT. Neither retrospective study establishes tradable alpha.

# %%
import json
from pathlib import Path
import pandas as pd
import financial_sentiment_utils as utils

bert, strategy, hashes = utils.load_evidence()
print(json.dumps(hashes, indent=2))

# %% [markdown]
# ## Observed feature packet
# These are derived features from 356 original FNSPID headlines for Agilent (A)
# and split-adjusted A/SPY opening prices. Training has 91 paired company-days;
# testing has 87. Text vocabulary/IDF and price scaling used training only.
# Vocabulary and raw articles are omitted; raw preprocessing is therefore not
# reproduced by this portable packet. The source implementation is included.

# %%
packet = json.loads(Path("data/historical_features.json").read_text())
display(pd.DataFrame([{
    "partition": part, "samples": len(values["samples"]),
    "first_entry": min(r["entry_at"] for r in values["samples"]),
    "last_outcome": max(r["exit_at"] for r in values["samples"]),
} for part, values in packet["partitions"].items()]))
print("Data kind:", packet["data_kind"])
print("Historical availability verified:", packet["historical_availability_verified"])

# %% [markdown]
# ## API contracts and units
# `refit_baselines()` fits training-mean, momentum, TF-IDF and combined Ridge
# controls and checks every prediction against the original sklearn solution.
# `model_comparison()` returns all BERT/baseline results in percentage points.
# `audit_strategy()` checks the daily curve against final PnL. Intraday maximum
# drawdown cannot be reconstructed from daily closes and is reported separately.
# `price_label()` computes raw and benchmark-adjusted price returns using aligned
# session arrays. These returns exclude cash dividends.

# %%
display(pd.DataFrame(utils.model_comparison(bert)))
display(pd.DataFrame(utils.audit_strategy(strategy)["models"]))

# %% [markdown]
# ## Timing example: arithmetic only
# The following prices are hand-written synthetic numbers used to explain the
# formula. They are not the real training packet and are never used as research
# performance evidence. Missing future outcomes raise an explicit error.

# %%
example = utils.price_label([100, 102, 101, 105], [100, 101, 101, 102], 0, 3)
assert abs(example["excess_return"] - 0.03) < 1e-12
print(example)
try:
    utils.price_label([100, 102], [100, 101], 0, 10)
except ValueError as error:
    print("Expected immature-label rejection:", error)

# %% [markdown]
# The historical experiments use date-only publication information, a two-session
# delay, outcome-maturity purging, and conservative event/source group isolation.
# These checks cannot reconstruct unknown historical receipts, certify causal
# attribution, or remove all dependence between overlapping test outcomes.
