# %% [markdown]
# # Price-label research: model refit and evidence review
# This notebook executes locally without API keys, market requests, model
# downloads, or trading. Every model and company is retained, including losses.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import financial_sentiment_utils as utils

# %% [markdown]
# ## 1. Refit four baselines on observed data
# The model is Ridge with alpha=10, fixed in the original protocol. Centering,
# intercept and model coefficients are estimated only on the 91 training rows.
# An independent NumPy dual solution is compared with all 87 saved predictions
# from sklearn LSQR. This is real numerical fitting, not a display of old scores.
# The prepared text transform is frozen and its vocabulary is not redistributed.

# %%
refitted = pd.DataFrame(utils.refit_baselines())
display(refitted)
assert len(refitted) == 4
assert refitted.max_prediction_difference.max() < 1e-6

# %% [markdown]
# ## 2. Review the separate 2023 nine-model experiment
# Three companies, 272 training rows, 154 held-out rows. BERT used 272 actual
# optimizer steps in the retained experiment. This cell loads its published
# receipt; it does not rerun the transformer or certify missing private archives.
# FinBERT measures text tone; the BERT regressor was supervised by future relative
# price change. A weak price label is not an independently annotated sentiment.

# %%
bert, strategy, hashes = utils.load_evidence()
all_models = pd.DataFrame(utils.model_comparison(bert))
display(all_models.pivot(index="model", columns="company", values="MAE_pp"))
display(all_models.pivot(index="model", columns="company", values="direction_accuracy"))
assert len(all_models) == 27

# %% [markdown]
# ## 3. Audit the separate 2026 trading comparison
# A: frozen five-minute price model. C: daily-refreshed 30-minute price control.
# B: C plus timestamp-audited text residuals. Each simulation started with $50,000
# on the same 32 sessions. The 2 bps per-side friction is a model assumption,
# not an actual broker commission. The B-C difference isolates this particular
# text adapter more closely than the B-A comparison.

# %%
audit = utils.audit_strategy(strategy)
display(pd.DataFrame(audit["models"]))
print("B minus C net simulated PnL:", audit["text_increment_B_minus_C"])
curves = pd.DataFrame(strategy["curves"]).set_index("date")
ax = curves.plot(figsize=(10, 4), title="Retrospective simulated equity: assumed 2 bps per side")
ax.set_ylabel("Simulated USD; not actual account equity")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Interpretation and attribution
# B loses money and underperforms C. C's tiny positive result at 2 bps becomes
# negative at 5 bps; lower turnover does not prove a durable directional signal.
# The 2023 BERT experiment does not feed the 2026 intraday strategy.
#
# The move-attribution module detects excess moves using only earlier volatility,
# retrieves verified preceding news, supports an unexplained output, and requires
# separately reviewed human labels before reporting false-attribution rates.
# A November 4 run detected 26 large moves over 1,123 tested historical intervals.
# None of the retrospective headlines has independently verified historical
# availability and relevance. Consequently no causal explanation is certified,
# and unavailable evaluation rates are not reported as zero.

# %%
import json
from pathlib import Path
attribution_path = Path("data/move_attribution_summary.json")
if attribution_path.is_file():
    display(json.loads(attribution_path.read_text()))
print("Research alpha established:", bert["alpha_established"])
print("Remaining: independent human attribution review, original 2023 raw-input recovery,")
print("new-period evaluation, student-recorded presentation, and course reviewer acceptance.")
