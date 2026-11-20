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
# # Decision-Aware Knowledge Base: Offline Build and Evaluation on EDC
#
# - This notebook runs the whole offline half on 10 decision records of
#   `eclipse-edc/Connector`, then checks each stage against the draft answer
#   keys in `specs/gold/` and tries the tuning knobs that need no API key
# - The pedagogical arc:
#   - Build the knowledge base: records, links, graph, staleness, gaps
#   - How well are records read?
#   - How well are records linked to code, and which signal helps?
#   - Which records look stale, and how the threshold changes that
#   - Which complex modules have no decision record
# - The answer keys were written by one person and are not reviewed yet, so
#   read the numbers as a first check, not a finding

# %%
# !pip install -q --root-user-action=ignore lizard==1.25.0 markdown-it-py==4.0.0 networkx==3.5 python-dotenv==1.2.4

import lizard
import markdown_it
import networkx

print("lizard version: ", lizard.version)
print("markdown-it-py version: ", markdown_it.__version__)
print("networkx version: ", networkx.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import pandas as pd

# %%
import helpers.hnotebook as hnotebook

import UmdTask745_MSML610_Fall2026_AI_Knowledge_Management_Development_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Make the `adrqa` package under `src/` importable.
utils.add_src_to_path()

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %%
import adrqa.config as aconfig
import adrqa.gitstore as agitsto
import adrqa.offline as aofflin

# %% [markdown]
# # Part 1: Build the Knowledge Base

# %% [markdown]
# ## Cell 1.1: Run the Offline Build
#
# - Same as `python -m adrqa build`: clone once, parse, link, build the graph,
#   score staleness, rank gaps, and write JSON to `data/`
# - Without an API key the replacement finder is the rule-based parser only
# - Scoring staleness follows each file's history, so this takes a little
#   while

# %%
result = aofflin.build_offline(log=_LOG.info)
print(
    f"records: {len(result['records'])}, links: {len(result['links'])}, "
    f"gap modules: {len(result['gaps'])}"
)

# %% [markdown]
# ## Cell 1.2: What Was Written
#
# - Each stage leaves one JSON file, so later stages and the second half of
#   the project read files instead of rerunning anything

# %%
for path in sorted((aconfig.DATA_DIR).glob("*.json")):
    print(f"{path.name:28s} {path.stat().st_size / 1024:8.1f} KB")

# %% [markdown]
# # Part 2: Reading the Records

# %% [markdown]
# ## Cell 2.1: The Parsed Records

# %%
records = utils.load_json("records.json")
display(utils.records_table(records))

# %% [markdown]
# ## Cell 2.2: Score Against the Extraction Key

# %%
display(utils.extraction_scores(records, utils.load_gold("extraction.json")))

# %% [markdown]
# - Everything is read correctly, but all 10 records follow one template, so
#   this says little about messier repositories
# - EDC records have no status line, so status is never scored

# %% [markdown]
# # Part 3: Linking Records to Code

# %% [markdown]
# ## Cell 3.1: Precision and Recall of the Links
#
# - A directory in the linking key stands for its non-test code files

# %%
links = utils.load_json("links.json")
gold_pairs = utils.expand_gold(
    utils.load_gold("linking.json"), agitsto.GitStore().tree
)
display(pd.DataFrame([utils.link_scores(links, gold_pairs)]))

# %% [markdown]
# - Nearly every right file is found (recall), but only about half of the
#   links are right (precision)
# - Wrong links come from names mentioned in passing and sentences like "this
#   will remain unchanged"

# %% [markdown]
# ## Cell 3.2: Switch Each Signal Off
#
# - Which signal earns its place? Drop one at a time and score the rest

# %%
display(utils.signal_ablation(links, gold_pairs))

# %% [markdown]
# - `explicit_class` carries the recall: without it recall collapses
# - `explicit_module` adds most of the wrong links: dropping it raises
#   precision but loses some right files, a trade-off worth tuning
# - `adr_commit` adds nothing on EDC

# %% [markdown]
# ## Cell 3.3: Why the Commit Signal Is Empty

# %%
display(pd.DataFrame(utils.load_json("adr_commit_signal.json")))

# %% [markdown]
# - The commits that added the records touched no code that still exists

# %% [markdown]
# # Part 4: Staleness

# %% [markdown]
# ## Cell 4.1: Scores Per Record

# %%
scores = utils.load_json("staleness.json")
threshold = aconfig.get_settings().stale_warn_threshold
display(utils.staleness_table(scores, threshold))

# %%
utils.plot_staleness(scores, threshold)

# %% [markdown]
# - Two records get no score (null), not 0: no linked code was found, so
#   there is nothing to measure
# - The count includes the commits that carried the decision out, so a high
#   score is a hint to check the record, not proof it is wrong

# %% [markdown]
# ## Cell 4.2: How the Threshold Changes the Warnings
#
# - The default of 15 is untested, so try a few values

# %%
display(utils.threshold_sweep(scores, [5, 10, 15, 25]))

# %% [markdown]
# # Part 5: Documentation Gaps

# %% [markdown]
# ## Cell 5.1: Complex Modules Without a Record

# %%
gaps = utils.load_json("gap_report.json")
display(pd.DataFrame(gaps).head(10))

# %%
utils.plot_gaps(gaps, top=10)

# %% [markdown]
# - Coverage only counts the 10 records used here, out of 113 in the
#   repository, so a module shown as uncovered may have a record we skipped

# %% [markdown]
# # Takeaways
#
# - The offline half turns loose Markdown records into a graph that answers
#   "which decisions govern this code?" and "what replaced this decision?"
#   without any model
# - Reading records is solved on this corpus; linking is the weak stage, with
#   high recall and low precision
# - Staleness and gaps are cheap hints that point a reviewer at records and
#   modules worth a look
# - The second half of the project (next PR) uses this graph to answer
#   questions with citations
