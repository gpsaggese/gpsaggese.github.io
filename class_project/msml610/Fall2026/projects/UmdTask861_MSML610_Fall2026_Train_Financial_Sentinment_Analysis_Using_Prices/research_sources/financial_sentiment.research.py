# %% [markdown]
# # Price-derived financial sentiment: reproducible development experiment
# 
# MSML610, Group 19. This notebook reruns a bounded real-data experiment offline. It
# does not establish alpha or complete the original prospective total-return study.
# The original API and example notebooks remain separate.
# 
# Research question: can a model trained on subsequent relative price returns improve
# on a price-only control or a pretrained text-tone control? Here the answer is not
# supported by the inspected development sample. The new FinBERT comparator was added
# after the original test results had been inspected; this is not a fresh blind test.
# 
# Run all cells with the installed research Python 3.11. Required retained inputs live
# under `reports/financial_sentiment_runs`; no network, credentials, installation or
# order submission is performed by this notebook. Each execution creates new outputs.

# %%
import hashlib, json, math, os, sys, tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image

repo = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "research/financial_sentiment_using_prices/run_historical.py").is_file())
research = repo / "research/financial_sentiment_using_prices"
sys.path.insert(0, str(research))
from run_historical import run, NY
from price_labels import _connected_groups
from finbert_comparison import prepare_packet, evaluate_tone_controls, validate_features

history = repo / "reports/financial_sentiment_runs/historical_development_20261021"
course = repo / "reports/financial_sentiment_runs/course_delivery_20261021"
registration = research / "comparison_protocol_20261021.json"
capture, original = history / "capture", history / "evaluation_verified"
packet_path = course / "tone_request/tone_request.json"
features_path = course / "tone_inference/tone_features.json"
required = [capture / "manifest.json", original / "run_manifest.json", packet_path, features_path, registration]
assert all(p.is_file() for p in required), "Restore the hash-bound private archive and cached tone receipts first"
output = Path(tempfile.mkdtemp(prefix="notebook_", dir=course))
(output / ".gitignore").write_text("*\n")
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
print("Offline research output:", output.relative_to(repo))
print("Python:", sys.version.split()[0])

# %% [markdown]
# ## Data and two distinct targets
# 
# FNSPID's immutable CSV prefix supplies original headlines and vendor ticker tags.
# Capture receipts are from 2026, while the reported source dates are historical.
# Exact historical versions, publication availability and independent relevance are
# not certified. A is Agilent Technologies, not an unspecified generic stock.
# 
# The separate retrospective target is
# `(A exit open / A entry open - 1) - (SPY exit open / SPY entry open - 1)`.
# Prices are split-adjusted and exclude cash dividends. Entry is the second following
# XNYS session after a date-only source date; exit is ten sessions after entry.
# Classes are -1 below -0.005, +1 above +0.005, otherwise 0. These weak price labels
# measure a later market outcome, not human judgments of positive or negative wording.
# 
# The original strict ten-session total-return track still requires publication,
# capture/availability and corporate-action evidence for stock and SPY. It has zero
# eligible labels here. This notebook does not replace that contract or backdate
# predictions. Event attribution or causal explanation is not inferred from returns.

# %%
base_report = run(capture, output / "baseline")
rows = read(output / "baseline/company_days.json")
split = read(output / "baseline/split.json")
assert base_report["headlines"] == 356
assert base_report["label_status_counts"]["RetrospectiveAvailable"] == 180
assert (len(split["train"]), len(split["test"])) == (91, 87)
assert len(split["excluded"]) == 2
assert all(set(r["reasons"]) == {"missing_prior_momentum_5", "missing_prior_momentum_20"}
           for r in split["excluded"])
assert base_report["strict_prospective_eligible_labels"] == 0
coverage = pd.DataFrame({"stage": ["original headlines", "observed company-day outcomes", "paired training", "paired test", "prior-price exclusions", "strict eligible"],
                         "count": [356, 180, 91, 87, 2, 0]})
display(coverage)
print("Excluded company-days:", [r["sample_id"] for r in split["excluded"]])
print("Observed bars:", base_report["price_bars"])

# %% [markdown]
# The 180 outcomes include two early company-days without the prior 5/20-session
# momentum inputs. Removing these same two rows for every model gives 178 paired rows
# (91 training + 87 test). This is a fair common comparison pool, not 180 test samples.
# Different headlines on one entry day are aggregated before splitting.

# %%
manifest = read(capture / "manifest.json")
calendar = read(capture / "calendar.json")["sessions"]
prices = {}
for request in manifest["requests"]:
    if request["file"].startswith("prices_"):
        for symbol, bars in read(capture / request["file"])["bars"].items():
            for bar in bars:
                day = datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(NY).date().isoformat()
                assert (symbol, day) not in prices
                prices[symbol, day] = bar["o"]
for row in rows:
    if row["label_status"] == "RetrospectiveAvailable":
        i, j = row["entry_session_index"], row["exit_session_index"]
        assert j-i == 10
        entry, exit_ = calendar[i]["date"], calendar[j]["date"]
        stock = prices[row["symbol"], exit_] / prices[row["symbol"], entry] - 1
        spy = prices["SPY", exit_] / prices["SPY", entry] - 1
        assert math.isclose(row["excess_return"], stock-spy, abs_tol=1e-12)
assert manifest["bars_summary"]["pagination_complete"] is True
print("Recomputed all 180 observed labels from archived opening prices; complete requested price pagination.")

# %% [markdown]
# ## Chronology, event groups and overlap
# 
# The fixed cutoff is 2019-01-01. All retained training outcomes must mature before
# that cutoff. Canonical URLs and conservative nearby-title proxies connect events;
# connected groups touching test are removed from training, including links through
# otherwise excluded rows. These proxies do not independently identify every event.
# There is no additional fixed-length embargo. Training-only outcome maturity purging
# does not remove overlapping ten-session labels within test. No IID significance or
# annualized Sharpe claim is made.

# %%
groups = _connected_groups(rows)
train_groups = {groups[r["sample_id"]] for r in split["train"]}
test_groups = {groups[r["sample_id"]] for r in split["test"]}
assert not train_groups.intersection(test_groups)
assert max(r["exit_at"] for r in split["train"]) < "2019-01-01T00:00:00Z"
assert min(r["entry_at"] for r in split["test"]) >= "2019-01-01T00:00:00Z"
display(pd.DataFrame([{ "partition": part, "rows": len(split[part]),
    "first_entry": min(r["entry_at"] for r in split[part]),
    "last_entry": max(r["entry_at"] for r in split[part]),
    "last_outcome": max(r["exit_at"] for r in split[part])} for part in ["train", "test"]]))
print("Shared proxy connected groups:", len(train_groups & test_groups), "; additional fixed embargo: none")

# %% [markdown]
# ## Six fixed controls, identical input budget
# 
# The original controls are training-mean return, prior-open momentum Ridge,
# headline TF-IDF Ridge and their combined model. Vocabulary, IDF, scaling and model
# parameters are fitted on training only. Ridge alpha is fixed at 10; there is no
# test-set hyperparameter search.
# 
# The additional comparator uses cached quantized `Xenova/finbert` at the registered
# immutable revision. Each distinct original headline receives the same input as
# TF-IDF, with no summaries or outcomes. Mean `P(positive)-P(negative)` is calibrated
# to returns with training-only StandardScaler/Ridge; a second model adds momentum.
# The softmax remains a tone score, not a probability that a stock rises. Model
# pretraining overlap and historical model availability are unverified. The frozen
# transformer itself is not fine-tuned in this project.

# %%
packet, features = read(packet_path), read(features_path)
regenerated_packet = prepare_packet(capture, output / "baseline", registration)
assert packet == regenerated_packet, "Different headline budget or split"
validate_features(packet, features)
assert features["any_truncated"] is False
comparison = evaluate_tone_controls(output / "baseline", packet_path, features_path, registration, output / "comparison")
table = pd.DataFrame(comparison["metrics"]).T
table["mae_pp"] = table["mae"] * 100
table["rmse_pp"] = table["rmse"] * 100
display(table[["samples", "mae_pp", "rmse_pp", "relative_direction_accuracy", "price_class_balanced_accuracy", "price_class_macro_f1"]])
print("Unique headlines / headline uses:", features["unique_headlines"], features["headline_uses"])
print("Model weights SHA256:", features["model"]["model_sha256"])
print("No model selection from the table; market predictive skill established:", comparison["market_predictive_skill_established"])

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(table.index, table["mae_pp"], color=["#666666", "#2864ae", "#438c68", "#67a47e", "#a57432", "#bb8a46"])
ax.set_xlabel("Mean absolute excess-price-return error (percentage points; lower is better)")
ax.set_title("A: inspected historical development test, 87 paired company-days")
ax.invert_yaxis(); fig.tight_layout()
fig.savefig(output / "mae_comparison.png", dpi=180)
display(Image(filename=str(output / "mae_comparison.png"))); plt.close(fig)

# %% [markdown]
# ## Cost scenarios without compounding overlapping labels
# 
# The 87 test labels overlap. Only 25 fixed, non-overlapping intervals are compounded;
# 62 overlapping rows are not compounded. All models use identical intervals. A
# positive relative-return prediction selects a long company position, otherwise
# cash; always-company, always-SPY and cash are controls. Positions close between
# intervals. This sparse, news-selected exposure is not a continuous daily portfolio.
# 
# Round-trip 0/10/20 bps are illustrative assumptions, divided equally between entry
# and exit, not verified broker charges. Prices exclude dividends and do not prove
# fills; cash interest, borrow, shorts and annualization are omitted. A large positive
# market return or near-always-long signal alone does not demonstrate alpha.

# %%
costs = comparison["cost_evaluation"]
assert costs["eligible_intervals"] == 25
assert costs["overlapping_test_rows_not_compounded"] == 62
assert costs["actual_broker_charges"] is None
cost_table = pd.DataFrame([{ "model": name, "assumed_roundtrip_bps": s["assumed_roundtrip_cost_bps"],
    "simulated_price_return_pct": 100*s["cumulative_simulated_price_return"], "roundtrips": s["roundtrip_trades"]}
    for name, scenarios in costs["models"].items() for s in scenarios])
display(cost_table.pivot(index="model", columns="assumed_roundtrip_bps", values="simulated_price_return_pct"))
fig, ax = plt.subplots(figsize=(9,5.5))
cost_table.pivot(index="model", columns="assumed_roundtrip_bps", values="simulated_price_return_pct").plot.barh(ax=ax)
ax.set_xlabel("Cumulative simulated price return (%)"); ax.set_ylabel("")
ax.set_title("25 sparse intervals; assumed round-trip cost scenarios")
ax.invert_yaxis(); ax.legend(title="Assumed bps", loc="lower right")
fig.tight_layout(); fig.savefig(output / "cost_scenarios.png", dpi=180)
display(Image(filename=str(output / "cost_scenarios.png"))); plt.close(fig)

# %% [markdown]
# ## Reproduction proof and limits
# 
# The frozen archive, input hashes, immutable model revision and complete requested
# price pagination make this calculation reproducible. Reproducibility cannot prove
# that a final historical headline version was available at its reported source date.
# Vendor ticker tags and the partial alphabetic source prefix introduce selection and
# coverage limitations. The test has been inspected and is a development sample.
# 
# The predeclared expansion is A, AMZN and MSFT. Within the fixed 32 MiB prefix, AMZN
# has 268 headlines from only 2020-04-27 through 2020-06-02 (26 company-days, zero
# training rows), while MSFT is absent. Neither supplies an expanded fitted result.
# These are source-coverage failures, not evidence that those companies had no news.
# 
# Next milestones are broader dated source coverage under a new source-budget
# registration, original total-return corporate-action certification, a licensed
# Loughran–McDonald control, predeclared horizon comparison, and independently
# annotated attribution with an explicit unexplained outcome. The course's LLM
# fine-tuning aspiration, Docker execution, formal course issue/PR and video remain
# unfinished. Negative results are reportable; a positive result is not required to
# justify changing a target after inspecting outcomes.

# %%
prediction_hashes = {name: sha(output / "baseline" / name) for name in
    ["predictions.json", "fitted_models.json", "company_days.json", "split.json"]}
assert all(value == sha(original / name) for name, value in prediction_hashes.items())
assert read(output / "comparison/predictions.json") == read(course / "comparison/predictions.json")
receipt = {"status": "Executed offline from archived real inputs", "baseline_hashes": prediction_hashes,
    "paired_train": 91, "paired_test": 87, "strict_eligible": 0, "market_predictive_skill_established": False,
    "tone_features_sha256": sha(features_path), "registration_sha256": sha(registration),
    "metrics": comparison["metrics"], "output": str(output.relative_to(repo))}
(output / "notebook_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ["status", "paired_train", "paired_test", "strict_eligible", "market_predictive_skill_established", "output"]}, indent=2))

# %% [markdown]
# ## Sources and delivery requirements
# 
# - [Official research idea](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/draft.Train_Financial_Sentinment_Analysis_Using_Prices.md)
# - [Class project guide](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/README.md)
# - [FNSPID pinned dataset](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/bf9189c41527198897d1af3e17b1a0095279fc45)
# - [FinBERT conversion revision](https://huggingface.co/Xenova/finbert/tree/8f269abebfdd9009d7d9b5e96af7e5c6bfe50b20)
# 
# See the research report's course gap table. Group 19 is a sign-up group, not an
# assumed GitHub issue number. Raw archives and cached model files stay private.
