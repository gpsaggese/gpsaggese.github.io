# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description
#
# This notebook is a short tour of the W&B Weave API. I use a toy problem,
# so every step runs in a few seconds and the API is the only thing to focus
# on: a keyword rule that labels short, made-up central bank sentences as
# hawkish, dovish, or neutral.
#
# It covers the five pieces of Weave that the project uses:
#
# 1. `weave.init`, to connect to a Weave project
# 2. `@weave.op`, to trace function calls
# 3. `weave.Dataset`, to store versioned evaluation data
# 4. `weave.Model`, to version a model together with its settings
# 5. `weave.Evaluation`, to score models on a dataset and compare them
#
# The real FOMC models (TF-IDF baseline, FOMC-RoBERTa, zero-shot) use the same
# API in `weave.example.ipynb`.
#
# References:
# - Weave documentation: https://weave-docs.wandb.ai/
# - Tracing: https://weave-docs.wandb.ai/guides/tracking/tracing
# - Models: https://weave-docs.wandb.ai/guides/core-types/models
# - Datasets: https://weave-docs.wandb.ai/guides/core-types/datasets
# - Evaluations: https://weave-docs.wandb.ai/guides/core-types/evaluations

# %% [markdown]
# ## Imports

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

# Third-party libraries.
import dotenv
import pandas as pd
import weave

# %%
# Use this for most notebooks.
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# Convert `display` into `print()`.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Configuration
#
# Weave needs a W&B API key. I keep mine in a `.env` file in the project
# folder (it is git-ignored), with one line: `WANDB_API_KEY=<key>`. The key is
# available at https://wandb.ai/authorize.

# %%
dotenv.load_dotenv(Path.cwd() / ".env")
assert os.environ.get("WANDB_API_KEY"), "Add WANDB_API_KEY to the .env file."

# Turn off the link Weave prints after every traced call, since this notebook
# makes dozens of calls and the Weave UI is easier to browse.
os.environ["WEAVE_PRINT_CALL_LINK"] = "false"
# Hide the HTTP requests to W&B that INFO-level logging would print (Weave
# uses both the `httpx` and `httpx2` clients).
for http_logger in ("httpx", "httpx2"):
    logging.getLogger(http_logger).setLevel(logging.WARNING)
# Stop the notebook logger from printing every Weave message a second time.
logging.getLogger("weave").propagate = False

# %% [markdown]
# ## 1. Connecting to a project with `weave.init`
#
# `weave.init` connects the notebook to a Weave project in my W&B account
# and creates the project the first time. From this point on, every call to a
# function decorated with `@weave.op` is recorded in that project.

# %%
weave_client = weave.init("msml610-weave-api")

# %% [markdown]
# ## 2. Tracing functions with `@weave.op`
#
# Decorating a function with `@weave.op` makes Weave record every call to it:
# the inputs, the output, how long it took, and any exception. When one op
# calls another, Weave nests the inner call under the outer one, so a trace
# is a tree that shows how the result was computed.
#
# My toy classifier has two ops:
#
# - `count_keyword_hits` counts how many keywords from a list appear in a
#   sentence. A keyword matches the start of a word, so "ease" matches "eased"
#   but not "increases".
# - `keyword_tone` calls it twice (hawkish words, dovish words) and picks the
#   label with more hits, or "neutral" on a tie
#
# The keyword lists are deliberately short. I improve them in section 4.

# %%
HAWKISH_WORDS = ["raise", "inflation"]
DOVISH_WORDS = ["lower", "cut"]


@weave.op
def count_keyword_hits(sentence: str, words: List[str]) -> int:
    """
    Count how many of the words appear in the sentence.

    :param sentence: text to search
    :param words: keywords to look for, matched against the start of each word
    :return: number of keywords found
    """
    tokens = re.findall(r"[a-z]+", sentence.lower())
    return sum(any(token.startswith(word) for token in tokens) for word in words)


@weave.op
def keyword_tone(sentence: str) -> Dict[str, Any]:
    """
    Label a sentence as hawkish, dovish, or neutral with a keyword rule.

    :param sentence: text to classify
    :return: the label and the number of hawkish and dovish keyword hits
    """
    hawkish_hits = count_keyword_hits(sentence, HAWKISH_WORDS)
    dovish_hits = count_keyword_hits(sentence, DOVISH_WORDS)
    if hawkish_hits > dovish_hits:
        label = "hawkish"
    elif dovish_hits > hawkish_hits:
        label = "dovish"
    else:
        label = "neutral"
    return {"label": label, "hawkish_hits": hawkish_hits, "dovish_hits": dovish_hits}


# %%
# Call the op like any Python function. This call and the two
# `count_keyword_hits` calls inside it now appear on the Traces page.
keyword_tone("The Committee decided to raise the target range to curb inflation.")

# %% [markdown]
# ## 3. Storing evaluation data with `weave.Dataset`
#
# A `weave.Dataset` is a named, versioned list of rows. Each row is a
# dictionary, and the keys become columns. Publishing the dataset stores it in
# the project and returns a reference. If I publish a changed version under
# the same name, Weave keeps both versions.
#
# The sentences below are short examples I wrote myself, in the style of FOMC
# statements, with a label I assigned to each one.

# %%
toy_rows = [
    {"sentence": "The Committee decided to raise the target range to curb inflation.", "label": "hawkish"},
    {"sentence": "Further rate increases may be appropriate to bring inflation down.", "label": "hawkish"},
    {"sentence": "Inflation remains elevated and the labor market is tight.", "label": "hawkish"},
    {"sentence": "Tighter policy is needed to keep inflation expectations anchored.", "label": "hawkish"},
    {"sentence": "The Committee decided to lower the target range to support employment.", "label": "dovish"},
    {"sentence": "Rate cuts may be needed if unemployment keeps rising.", "label": "dovish"},
    {"sentence": "Economic activity has weakened and inflation pressures have eased.", "label": "dovish"},
    {"sentence": "Accommodative policy will remain in place to support the recovery.", "label": "dovish"},
    {"sentence": "The Committee will continue to monitor incoming data.", "label": "neutral"},
    {"sentence": "The meeting was held in Washington on Tuesday.", "label": "neutral"},
    {"sentence": "Household spending grew at a moderate pace.", "label": "neutral"},
    {"sentence": "The Committee will keep the target range unchanged.", "label": "neutral"},
]

toy_dataset = weave.Dataset(name="toy-tone-sentences", rows=toy_rows)
toy_dataset_ref = weave.publish(toy_dataset)

print("Dataset reference:", toy_dataset_ref.uri())

# %%
# Load the dataset back through its reference, the way a teammate or a later
# session gets exactly the same rows.
loaded_dataset = toy_dataset_ref.get()
display(pd.DataFrame(list(loaded_dataset.rows)))

# %% [markdown]
# ## 4. Versioning a model with `weave.Model`
#
# A `weave.Model` bundles a `predict` op with the settings it depends on. The
# settings are declared as class attributes, like a dataclass. When I change
# a setting, Weave stores the result as a new version of the model, and every
# trace and evaluation points to the exact version that produced it.
#
# Below, the keyword rule becomes a model whose keyword lists are attributes.

# %%
class KeywordToneModel(weave.Model):
    """
    Label sentences as hawkish, dovish, or neutral by counting keywords.
    """

    hawkish_words: List[str]
    dovish_words: List[str]

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Label one sentence.

        :param sentence: text to classify
        :return: the label and the number of hawkish and dovish keyword hits
        """
        hawkish_hits = count_keyword_hits(sentence, self.hawkish_words)
        dovish_hits = count_keyword_hits(sentence, self.dovish_words)
        if hawkish_hits > dovish_hits:
            label = "hawkish"
        elif dovish_hits > hawkish_hits:
            label = "dovish"
        else:
            label = "neutral"
        return {
            "label": label,
            "hawkish_hits": hawkish_hits,
            "dovish_hits": dovish_hits,
        }


# %%
# Use the keyword lists of the plain function above for version 1.
model_v1 = KeywordToneModel(
    name="keyword-tone",
    hawkish_words=HAWKISH_WORDS,
    dovish_words=DOVISH_WORDS,
)

# Fix the two problems of version 1 in version 2: drop "inflation", which
# appears in hawkish and dovish sentences alike, and add more words to each
# list.
model_v2 = KeywordToneModel(
    name="keyword-tone",
    hawkish_words=["raise", "increase", "tight", "elevated"],
    dovish_words=["lower", "cut", "support", "weak", "ease", "accommodative"],
)

# Publish both under the same name to get two versions of one model.
model_v1_ref = weave.publish(model_v1)
model_v2_ref = weave.publish(model_v2)
print("Version 1:", model_v1_ref.uri())
print("Version 2:", model_v2_ref.uri())

# %%
# Compare the two versions on one dovish sentence: version 1 is misled by
# "inflation", while version 2 picks up "weakened" and "eased".
example_sentence = "Economic activity has weakened and inflation pressures have eased."
print("Version 1:", model_v1.predict(example_sentence))
print("Version 2:", model_v2.predict(example_sentence))

# %% [markdown]
# ## 5. Scoring models with `weave.Evaluation`
#
# An evaluation runs a model on every row of a dataset and scores each output.
# Weave matches dataset columns to argument names: the `sentence` column goes
# to `predict(sentence)`, and the `label` column goes to every scorer that has
# a `label` argument. The model's return value reaches the scorers as
# `output`.
#
# Weave supports two kinds of scorers:
#
# - **A function decorated with `@weave.op`**, for scores that make sense
#   row by row. Weave summarizes them automatically; for `True`/`False` values
#   it reports the fraction that are `True`.
# - **A subclass of `weave.Scorer`**, when the summary needs custom logic.
#   `score()` runs once per row, and `summarize()` receives all of the row
#   scores at the end.
#
# I use one of each: exact match (which gives the accuracy) and recall per
# class, which tells me which kind of sentence the model gets wrong.

# %%
@weave.op
def exact_match(label: str, output: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check whether the predicted label matches the true label.

    :param label: true label from the dataset
    :param output: the model's prediction
    :return: whether the prediction is correct
    """
    return {"correct": output["label"] == label}


class PerClassRecall(weave.Scorer):
    """
    Compute the share of sentences of each class that the model labels correctly.
    """

    @weave.op
    def score(self, *, output: Dict[str, Any], label: str) -> Dict[str, Any]:
        """
        Record the true label and whether the prediction matches it.

        :param output: the model's prediction
        :param label: true label from the dataset
        :return: the true label and whether the prediction is correct
        """
        return {"label": label, "correct": output["label"] == label}

    @weave.op
    def summarize(self, score_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Turn the per-row results into a recall for each class.

        :param score_rows: the outputs of `score()` for every row
        :return: recall per class
        """
        scores = pd.DataFrame([row for row in score_rows if row])
        recall = scores.groupby("label")["correct"].mean()
        return {f"recall_{label}": float(value) for label, value in recall.items()}


toy_evaluation = weave.Evaluation(
    name="toy-tone-evaluation",
    dataset=toy_dataset,
    scorers=[exact_match, PerClassRecall()],
)

# %%
# Run each evaluation with `await`, since `evaluate()` is async. Name each run
# with `display_name` so the two versions are easy to tell apart on the Evals
# page.
summary_v1 = await toy_evaluation.evaluate(
    model_v1, __weave={"display_name": "keyword-tone-v1"}
)
summary_v2 = await toy_evaluation.evaluate(
    model_v2, __weave={"display_name": "keyword-tone-v2"}
)

# %%
# Flatten the nested summaries into one table to compare the two versions.
comparison = pd.DataFrame(
    {
        version: {
            "accuracy": summary["exact_match"]["correct"]["true_fraction"],
            **summary["PerClassRecall"],
            "mean_latency_s": summary["model_latency"]["mean"],
        }
        for version, summary in [("v1", summary_v1), ("v2", summary_v2)]
    }
).round(3)
display(comparison)

# %% [markdown]
# ## 6. Inspecting everything in the Weave UI
#
# All of the above lives in the project page that `weave.init` printed:
#
# - **Traces**: every op call. Opening a `keyword_tone` call shows the two
#   nested `count_keyword_hits` calls, with their inputs and outputs.
# - **Datasets**: `toy-tone-sentences`, with its rows and versions.
# - **Models**: `keyword-tone`, with two versions. Comparing them shows which
#   attributes changed (the keyword lists).
# - **Evals**: the two evaluation runs. Selecting both and clicking
#   **Compare** shows the summary metrics side by side, plus every sentence
#   with both predictions, so I can filter to the sentences where the
#   versions disagree.
#
# This is the same workflow that `weave.example.ipynb` applies to the real
# FOMC models.
#
# One caveat about the numbers: I chose the version 2 keywords by looking at
# these same 12 sentences, so its perfect score says nothing about new text.
# That is why `weave.example.ipynb` never tunes anything on the test set.
