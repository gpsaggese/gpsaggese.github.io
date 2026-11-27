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
# # `adrqa` API: Building a Decision Graph From Architecture Decision Records
#
# - This notebook walks through the offline API of `adrqa` one building block
#   at a time, on the decision records (ADRs) of `eclipse-edc/Connector` at one
#   pinned commit, with no API key
# - The pedagogical arc:
#   - Read the repository at a pinned commit without checking it out
#   - Parse one record into fields, references and replacements
#   - Check a language model's replacement claims against the record text
#   - Link the record to the code it governs
#   - Build the decision graph and query it
#   - Score staleness and rank documentation gaps

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
import adrqa.gaps as agaps
import adrqa.gitstore as agitsto
import adrqa.graph as agraph
import adrqa.linker as alinker
import adrqa.parser as aparser
import adrqa.relations as arelati
import adrqa.staleness as astalen

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**: a repository keeps its design decisions in
#   Markdown records, but nothing ties a record to the code it governs, to the
#   record that replaced it, or says whether it still matches the code
# - **Key abstraction**: a decision graph whose nodes are records and files,
#   and whose edges are `governs`, `supersedes` and `references`
# - **Mental model**:
#
# | Object | Description | Output |
# | :----- | :---------- | :----- |
# | `GitStore` | Reads git objects at the pinned commit | file text, history |
# | `parse_record()` | Markdown record to labelled pieces, no model | record dict |
# | `model_edges()` | Optional model finds replacements, then checked | edges |
# | `Linker` | Names in the text to files in the tree | link dicts |
# | `build_graph()` | Records, files and the three edge kinds | `networkx` graph |
# | `score_record()` | Later code commits / (1 + record edits) | staleness |
# | `rank_gaps()` | Complexity x (1 - coverage) per module | gap list |
#
# - The `adrqa build` command runs all of them in this order and writes the
#   results to `data/`, see the example notebook

# %% [markdown]
# # Part 1: Read the Repository at a Pinned Commit

# %% [markdown]
# ## Cell 1.1: Clone Once and Pin the Commit
#
# - The corpus is fixed: one repository, one commit SHA, ten records, so every
#   run reads exactly the same files
# - `GitStore` never checks files out: it reads git objects with `git show`
#   and `git cat-file`, which also avoids the Windows path length limit
# - The first run clones the repository into `.cache/` (about 70 MB)

# %%
# Clone the repository if needed and open it at the pinned commit.
agitsto.GitStore.ensure_clone()
store = agitsto.GitStore()
print("repository:", aconfig.REPO_URL)
print("pinned commit:", store.sha[:9])
print("paths in the tree:", len(store.tree))
print("records used:", len(aconfig.RECORDS))

# %% [markdown]
# ## Cell 1.2: Read One Decision Record
#
# - We follow one record through the whole notebook: the 2026-09-22 record on
#   per-participant resource ids, because it partly replaces an older record

# %%
record_id = "2026-09-22-per-participant-resource-ids"
path = f"{aconfig.RECORD_ROOT}{record_id}/README.md"
text = store.read(path)
print("\n".join(text.splitlines()[:22]))

# %% [markdown]
# # Part 2: Parse a Record Without a Model

# %% [markdown]
# ## Cell 2.1: Fields, References and Replacements
#
# - `parse_record()` uses `markdown-it-py` to find the title and cut the text
#   at its `##` headings into Decision, Rationale and Approach
# - Links of the form `../<record>/README.md` become references
# - A sentence with "supersedes" and a link becomes a replacement edge; a
#   quoted statement in that sentence makes it a partial replacement

# %%
corpus = set(aconfig.RECORDS)
record = aparser.parse_record(record_id, text, corpus)
print("title:", record["title"])
print("date:", record["date"])
print("status:", record["status"])
print("missing fields:", record["missing_fields"])
print("references:", record["references"])
display(pd.DataFrame(record["supersedes"]))

# %% [markdown]
# ## Cell 2.2: One Field
#
# - Each field keeps its Markdown, so later steps can quote it exactly

# %%
print(record["fields"]["rationale"][:600])

# %% [markdown]
# # Part 3: Check a Model's Replacement Claims

# %% [markdown]
# ## Cell 3.1: Keep Only Claims the Record Supports
#
# - The rule parser only understands EDC's own wording; on six other ADR
#   repositories it found 0 of 7 replacements
# - With an API key, `relations.py` asks a chat model instead, then keeps a
#   claim only if its target is a real record and its evidence quote appears
#   word for word in the record
# - Here a scripted fake model makes one true claim and two invented ones, so
#   we can see the checks work without a key

# %%
true_claim = {
    "type": "supersedes",
    "target": "2025-08-26-participant-context",
    "scope": "partial",
    "evidence": "It supersedes the statement of the [participant context]"
    "(../2025-08-26-participant-context/README.md) decision record",
    "replaced": "the store and service interface for those entities won't change",
}
invented_target = {**true_claim, "target": "2023-01-01-made-up-record"}
invented_quote = {
    **true_claim,
    "target": "2025-05-28-dataspace-profile-context",
    "evidence": "This replaces the dataspace profile context record.",
}
chat = utils.ScriptedChat([true_claim, invented_target, invented_quote])
edges = arelati.model_edges(record_id, text, corpus, chat)
display(pd.DataFrame(edges)[["newer", "older", "scope", "statement"]])

# %% [markdown]
# - Only the true claim survives: the invented target is not in the corpus,
#   and the invented quote is not in the record
# - In `build`, model edges are merged with the rule edges, so a model miss
#   can never remove an edge the rules found

# %% [markdown]
# # Part 4: Link the Record to Code

# %% [markdown]
# ## Cell 4.1: Names in the Text
#
# - `Linker` pulls tokens out of the text and resolves each against the tree:
#   - `explicit_class`: a CamelCase name that matches a `.java` file
#   - `explicit_module`: a kebab-case name that matches a directory
#   - `explicit_path`: a path in backticks
#   - `explicit_file`: a file name that is unique in the tree

# %%
linker = alinker.Linker(store)
links = linker.link_record(record_id, text)
links_df = pd.DataFrame(links)
display(links_df["signal"].value_counts())
display(links_df[["signal", "evidence", "file_id"]].head(8))

# %% [markdown]
# ## Cell 4.2: Files in the Commit That Added the Record
#
# - The second signal is the commit that added the record: if code changed in
#   the same commit, that code likely implements the decision

# %%
commit_links, report = linker.adr_commit_links(record_id, record["path"])
print(report)
print("links from the commit signal:", len(commit_links))

# %% [markdown]
# - On EDC this signal finds nothing: those commits touch only the record

# %% [markdown]
# # Part 5: Build and Query the Decision Graph

# %% [markdown]
# ## Cell 5.1: Parse and Link All Ten Records

# %%
texts = store.read_many(
    [f"{aconfig.RECORD_ROOT}{r}/README.md" for r in aconfig.RECORDS]
)
text_of = {
    r: texts[f"{aconfig.RECORD_ROOT}{r}/README.md"] for r in aconfig.RECORDS
}
records = [aparser.parse_record(r, text_of[r], corpus) for r in aconfig.RECORDS]
all_links = [
    l for r in aconfig.RECORDS for l in linker.link_record(r, text_of[r])
]
graph = agraph.build_graph(records, all_links)
kinds = pd.Series(
    [d["kind"] for _, _, d in graph.edges(data=True)]
).value_counts()
print("nodes:", graph.number_of_nodes())
display(kinds)

# %% [markdown]
# ## Cell 5.2: Ask the Graph Questions
#
# - Which records govern a directory?
# - Which newer record replaces an older one, and which statement?

# %%
directory = "spi/control-plane-spi/src/main/java/org/eclipse/edc/protocol/spi"
print("records governing", directory)
for rid in agraph.governing_records(graph, directory):
    print("  ", rid)

# %%
for newer, edge in agraph.superseded_by(graph, "2025-08-26-participant-context"):
    print(f"{newer} replaces part of 2025-08-26-participant-context")
    print("statement no longer true:", edge["statement"])

# %% [markdown]
# # Part 6: Staleness and Gaps

# %% [markdown]
# ## Cell 6.1: Is the Record Still Current?
#
# - `stale = later commits to the linked code / (1 + later edits to the record)`
# - Shared build files are linked but left out of the commit count, since
#   almost every commit touches them
# - Names the record uses that no longer exist in the code are reported too

# %%
files = sorted({l["file_id"] for l in all_links if l["record_id"] == record_id})
dead = astalen.dead_identifiers(store, text, linker.classes, linker.dirs_by_name)
score = astalen.score_record(store, record, files, dead)
print(score)
threshold = aconfig.get_settings().stale_warn_threshold
for warning in astalen.warnings_for(score, threshold, record):
    print("-", warning["plain"])

# %% [markdown]
# - No warning: the score is below the default threshold of 15

# %% [markdown]
# ## Cell 6.2: Complex Code No Record Covers
#
# - `lizard` measures cyclomatic complexity per function; it is summed per
#   module and multiplied by the share of the module's files that no record
#   links to

# %%
governed = {l["file_id"] for l in all_links}
gaps = agaps.rank_gaps(store, governed, top=5)
display(pd.DataFrame(gaps))
