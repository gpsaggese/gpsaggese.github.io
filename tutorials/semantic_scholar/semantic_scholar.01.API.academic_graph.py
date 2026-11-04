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
# # Semantic Scholar Academic Graph API
#
# - This notebook teaches the Semantic Scholar Academic Graph API with plain
#   `requests` calls on 2 well-known papers: "Attention Is All You Need"
#   (`ARXIV:1706.03762`) and "BERT" (`ARXIV:1810.04805`)
# - The pedagogical arc:
#   - Overview: the 3 services, the base URL, and the objects of the graph
#   - A paper by id, and the `fields=` parameter
#   - Search, with one page change
#   - Batch lookup of many papers in one request
#   - Authors
#   - Citations and references
#   - Errors and limits: `404`, `429`, and the retry
#   - A minimal end to end flow
#   - Optional: the Recommendations API
# - The notebook runs without an API key
#   - Without a key, a call can get `429` from the pool shared by all users
#   - The notebook retries and then prints a clear message instead of crashing

# %%
# !pip install -q requests==2.34.2 graphviz==0.21

import graphviz
import requests

print("requests version: ", requests.__version__)
print("graphviz version: ", graphviz.__version__)

# %%
# %load_ext autoreload
# %autoreload 2

import inspect
import logging

import pandas as pd

# %%
import helpers.hnotebook as hnotebook

import semantic_scholar_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: Overview

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**: find scientific papers, their authors, and the
#   citations between them, with one HTTP API
#   - The data comes from Semantic Scholar, a free search engine run by Ai2
#   - The API returns JSON
# - **Key abstraction**: a graph of papers, authors, and citations
#   - Every paper has several ids, and the API accepts all of them
#   - Every call returns only the fields that you ask for with `fields=`
# - **Mental model**: see the table below
# - **Key endpoints**: all under the base URL of the Academic Graph API
#   - `GET /paper/{id}`, `GET /paper/search`, `POST /paper/batch`
#   - `GET /author/search`, `GET /author/{id}`, `GET /author/{id}/papers`
#   - `GET /paper/{id}/citations`, `GET /paper/{id}/references`
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | Academic Graph API | Papers, authors, citations, venues | Core of this notebook |
# | Recommendations API | Papers similar to a given paper | Last part, optional |
# | Datasets API | Bulk download of the whole corpus | Only named here, not used |
# | `Paper` | One paper | Id: `paperId`, `DOI:<doi>`, or `ARXIV:<id>` |
# | `Author` | One author profile | Id: `authorId`, a string of digits |
# | `Citation` | A link from a citing paper to a cited paper | Wrapper keys `citingPaper` and `citedPaper` |
# | `Venue` | The journal or conference of a paper | A field of `Paper`, no endpoint |
#
# - Rules for every call
#   - The key goes only in the header `x-api-key`, never in the URL
#   - Without `fields=`, a paper has only `paperId` and `title`
#   - The official spec is at `https://api.semanticscholar.org/graph/v1/swagger.json`
# - TODO(gp): Check the anchors of the links to the official docs page below, since
#   the page answered `429` and could not be opened when this notebook was written
#   - The endpoint names, the parameters, and the limits in this notebook come from
#     the official spec above

# %% [markdown]
# ## Cell 1.1: The base URL and the API key
#
# - The Academic Graph API and the Recommendations API have each a base URL
# - The key is optional: read it from the environment variable
#   `SEMANTIC_SCHOLAR_API_KEY`
#   - The notebook only checks if the variable is set, it never prints the key
# - Every call in this notebook goes through `utils.call_api()`
#   - It is a thin wrapper of a plain `requests.request()` call
#   - It adds the header `x-api-key` when a key exists, waits 1 second after each
#     call, and retries on `429` and `5xx`
#   - Cell 7.3 prints its source

# %%
# Print the base URL of each service and check if the environment holds a key.
print("GRAPH_URL=", utils.GRAPH_URL)
print("RECOMMENDATIONS_URL=", utils.RECOMMENDATIONS_URL)
has_api_key = utils.cell1_1_has_api_key()
print("has_api_key=", has_api_key)
# Outcome: 2 URLs, then `True` if the key is set, else `False`. Both work.

# %% [markdown]
# ## Cell 1.2: The objects of the graph
#
# - A `Paper` is linked to its `Author` profiles, and to its `Venue`
# - A `Citation` links a citing paper to a cited paper
#   - The API has no endpoint for a venue: it is a field of a paper

# %%
# Draw the 4 objects of the graph and the keys that link them.
diagram = utils.cell1_2_get_object_diagram()
display(diagram)
# Outcome: a diagram with 4 boxes: `Author`, `Paper`, `Venue`, and `Citation`.

# %% [markdown]
# # Part 2: A paper by id

# %% [markdown]
# ## Primitive 1: `GET /paper/{id}`
#
# - **Mental model**: one call returns one paper, chosen by an id
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `GET /paper/{id}` | One paper by id | Answer is a JSON object |
# | `{id}` | Any id form of the paper | `paperId`, `DOI:<doi>`, `ARXIV:<id>`, and more |
# | `fields=a,b` | Fields to return | Default: only `paperId` and `title` |
# | `authors` | List of author objects | Each has `authorId` and `name` |
# | `externalIds` | Ids in other systems | `DOI`, `ArXiv`, `DBLP`, `CorpusId`, ... |

# %% [markdown]
# ## Cell 2.1: The default answer has few fields

# %%
# Call `GET /paper/{id}` with no `fields`, using the arXiv id of the paper.
attention_id = "ARXIV:1706.03762"
response = utils.call_api("GET", f"{utils.GRAPH_URL}/paper/{attention_id}")
paper = response.json()
print("status_code=", response.status_code)
print("paper=", paper)
# Outcome: status 200, and a dict with only 2 keys: `paperId` and `title`.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper

# %% [markdown]
# ## Cell 2.2: The parameter `fields` selects more

# %%
# Ask for 8 fields, as a comma separated list.
fields = ",".join(
    [
        "title",
        "year",
        "authors",
        "externalIds",
        "openAccessPdf",
        "citationCount",
        "publicationDate",
        "publicationTypes",
    ]
)
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/{attention_id}",
    params={"fields": fields},
)
paper = response.json()
print("keys=", sorted(paper))
print("year=", paper.get("year"))
print("citationCount=", paper.get("citationCount"))
print("publicationDate=", paper.get("publicationDate"))
print("publicationTypes=", paper.get("publicationTypes"))
print("externalIds=", paper.get("externalIds"))
print("openAccessPdf=", paper.get("openAccessPdf"))
print("authors[:2]=", paper.get("authors", [])[:2])
# Outcome: the 8 fields and `paperId`. `year` is an integer, `citationCount` is an
# integer larger than 0, and each author has `authorId` and `name`.

# %% [markdown]
# ## Cell 2.3: Three id forms reach the same paper
#
# - The id can be the `paperId` of Semantic Scholar, `DOI:<doi>`, or `ARXIV:<id>`
# - The full list of id forms is in the official docs of the endpoint

# %%
# Look up the same paper with 3 id forms, and compare the `paperId` of each answer.
id_forms = [
    "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    "DOI:10.48550/arXiv.1706.03762",
    "ARXIV:1706.03762",
]
paper_ids = []
for id_form in id_forms:
    response = utils.call_api("GET", f"{utils.GRAPH_URL}/paper/{id_form}")
    paper_ids.append(response.json().get("paperId"))
print("paper_ids=", paper_ids)
print("same paper=", len(set(paper_ids)) == 1)
# Outcome: the same `paperId` three times, so `True`.

# %% [markdown]
# # Part 3: Search

# %% [markdown]
# ## Primitive 2: `GET /paper/search`
#
# - **Mental model**: a search returns one page of a long list, plus the keys to
#   walk to the next page
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `query` | Plain text to search | No query syntax, and hyphens give no match |
# | `limit` | Size of the page | At most 100 |
# | `offset` | Position of the first result | Start of the page |
# | `fields` | Fields of each paper | Same rule as for one paper |
# | `total` | Number of matches | Integer in the answer |
# | `offset`, `next` | Position of this page, and of the next page | `next` is absent on the last page |
# | `data` | List of papers of the page | Empty list if nothing matches |
#
# - The search returns at most 1,000 relevance ranked results
#   - For more, the docs name the endpoint `/paper/search/bulk` and the Datasets API

# %% [markdown]
# ## Cell 3.1: The first page

# %%
# Search a text, and ask for a page of 3 papers with 2 fields.
search_params = {
    "query": "attention is all you need",
    "limit": 3,
    "fields": "title,year",
}
response = utils.call_api(
    "GET", f"{utils.GRAPH_URL}/paper/search", params=search_params
)
page1 = response.json()
print("keys=", sorted(page1))
print("total=", page1.get("total"))
print("offset=", page1.get("offset"))
print("next=", page1.get("next"))
display(pd.DataFrame(page1.get("data", [])))
# Outcome: keys `data`, `next`, `offset`, `total`. `total` is an integer larger
# than 3, `offset` is 0, `next` is 3, and the table has 3 papers.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_relevance_search

# %% [markdown]
# ## Cell 3.2: One page change

# %%
# Get the next page: send the value `next` of page 1 as the `offset`.
search_params["offset"] = page1.get("next", 3)
response = utils.call_api(
    "GET", f"{utils.GRAPH_URL}/paper/search", params=search_params
)
page2 = response.json()
print("offset=", page2.get("offset"))
print("next=", page2.get("next"))
display(pd.DataFrame(page2.get("data", [])))
# Outcome: `offset` is 3, `next` is 6, and the 3 papers differ from page 1.

# %% [markdown]
# # Part 4: Batch lookup

# %% [markdown]
# ## Primitive 3: `POST /paper/batch`
#
# - **Mental model**: one request looks up many papers, so it saves calls
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `POST /paper/batch` | Many papers in one call | At most 500 ids per call |
# | JSON body `{"ids": [...]}` | The ids to look up | Any id form, mixed |
# | `fields=` | Fields of each paper | In the query string, not in the body |
# | Answer | A JSON list | Same order as `ids`; an unknown id gives `null` |

# %% [markdown]
# ## Cell 4.1: Two papers and one unknown id

# %%
# Look up 3 ids in one call: 2 real papers and 1 id that does not exist.
batch_ids = ["ARXIV:1706.03762", "ARXIV:1810.04805", "ARXIV:0000.00000"]
response = utils.call_api(
    "POST",
    f"{utils.GRAPH_URL}/paper/batch",
    params={"fields": "title,year"},
    json_body={"ids": batch_ids},
)
papers = response.json()
print("status_code=", response.status_code)
print("papers=", papers)
# Outcome: status 200 and a list of 3 entries in the order of `batch_ids`. The
# third entry is `None`, since the id is unknown.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers

# %% [markdown]
# # Part 5: Authors

# %% [markdown]
# ## Primitive 4: `Author`
#
# - **Mental model**: an author is a profile with an `authorId`, and the profile
#   owns a list of papers
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `GET /author/search` | Find authors by name | `query`, `limit`, `offset`, `fields` |
# | `GET /author/{id}` | One author profile | Fields: `name`, `paperCount`, `hIndex`, ... |
# | `GET /author/{id}/papers` | Papers of one author | A page: `offset`, `next`, `data` |
# | `authorId` | Id of a profile | A string of digits |

# %% [markdown]
# ## Cell 5.1: Search an author by name

# %%
# Search the profiles that match a name.
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/author/search",
    params={"query": "Ashish Vaswani", "limit": 3, "fields": "name,paperCount"},
)
author_page = response.json()
print("keys=", sorted(author_page))
display(pd.DataFrame(author_page.get("data", [])))
# Outcome: keys `data`, `offset`, `total`. The table has `authorId`, `name`, and
# `paperCount`. One person can have more than one profile.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Author-Data/operation/get_graph_get_author_search

# %% [markdown]
# ## Cell 5.2: One author by id
#
# - The `authorId` below is the one of the first author in the output of Cell 2.2

# %%
# Get one profile by its `authorId`.
author_id = "40348417"
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/author/{author_id}",
    params={"fields": "name,paperCount,hIndex"},
)
author = response.json()
print("author=", author)
# Outcome: a dict with `authorId`, `name`, `paperCount`, and `hIndex`. The counts
# are integers larger than 0.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Author-Data/operation/get_graph_get_author

# %% [markdown]
# ## Cell 5.3: The papers of one author

# %%
# List the first 3 papers of the profile.
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/author/{author_id}/papers",
    params={"limit": 3, "fields": "title,year"},
)
author_papers = response.json()
print("keys=", sorted(author_papers))
display(pd.DataFrame(author_papers.get("data", [])))
# Outcome: keys `data`, `next`, `offset`. The table has 3 papers.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Author-Data/operation/get_graph_get_author_papers

# %% [markdown]
# # Part 6: Citations and references

# %% [markdown]
# ## Primitive 5: `Citation`
#
# - **Mental model**: a citation is a link between 2 papers, and each item of the
#   answer wraps the paper at the other end of the link
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `GET /paper/{id}/citations` | Papers that cite this paper | Items have the key `citingPaper` |
# | `GET /paper/{id}/references` | Papers that this paper cites | Items have the key `citedPaper` |
# | `limit`, `offset` | Page control | `limit` at most 1000 |
# | `fields=` | Fields of the wrapped paper | Also `contexts` and `isInfluential` on the link |

# %% [markdown]
# ## Cell 6.1: Who cites the paper

# %%
# List 3 papers that cite "Attention Is All You Need".
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/{attention_id}/citations",
    params={"limit": 3, "fields": "title,year"},
)
citations = response.json()
citation_items = citations.get("data", [])
print("keys=", sorted(citations))
print("item keys=", [sorted(item) for item in citation_items[:1]])
display(pd.DataFrame([item["citingPaper"] for item in citation_items]))
# Outcome: keys `data`, `next`, `offset`. Each item has the single key
# `citingPaper`, and the table has 3 papers.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper_citations

# %% [markdown]
# ## Cell 6.2: What the paper cites

# %%
# List 3 papers that "Attention Is All You Need" cites.
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/{attention_id}/references",
    params={"limit": 3, "fields": "title,year"},
)
references = response.json()
reference_items = references.get("data", [])
print("keys=", sorted(references))
print("item keys=", [sorted(item) for item in reference_items[:1]])
display(pd.DataFrame([item["citedPaper"] for item in reference_items]))
# Outcome: keys `data`, `next`, `offset`. Each item has the single key
# `citedPaper`, and the table has 3 papers.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_get_paper_references
# - The answer of `/references` can also hold the key `citingPaperInfo`
#   - TODO(gp): Check this key, since it is not in the official spec

# %% [markdown]
# # Part 7: Errors and limits

# %% [markdown]
# ## Primitive 6: HTTP status codes
#
# - **Mental model**: the status code tells if to fix the request, or to wait
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | `200` | The call worked | The body is the JSON answer |
# | `404` | The paper or author does not exist | Body has the key `error`; do not retry |
# | `429` | Too many requests | Wait, then retry; body has `message` and `code` |
# | `5xx` | Error of the server | Wait, then retry |
# | Retry schedule | Wait 2, 4, 8, then 16 seconds | `utils.BACKOFF_SEC` |
# | Throttle | Wait 1 second after each call | A key allows 1 request per second |
#
# - Without a key, all users share one pool, so `429` can come at any time
# - A free key has its own limit
#   - Ask for a key with the form at https://www.semanticscholar.org/product/api
#   - TODO(gp): Check the current limit per endpoint of a key, and of the shared
#     pool, since the docs page could not be opened

# %% [markdown]
# ## Cell 7.1: A missing paper gives 404

# %%
# Ask for an arXiv id that does not exist.
response = utils.call_api("GET", f"{utils.GRAPH_URL}/paper/ARXIV:0000.00000")
print("status_code=", response.status_code)
print("body=", response.json())
# Outcome: status 404 and a body with the key `error`. The helper does not retry
# a `404`.

# %% [markdown]
# ## Cell 7.2: The shared pool can give 429
#
# - This cell sends one call with no key and no retry
#   - Run it a few times to see both outcomes

# %%
# Send one call without the key and without a retry, to see the raw answer.
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/{attention_id}",
    params={"fields": "title"},
    use_key=False,
    backoff_sec=(),
)
print("status_code=", response.status_code)
print("body=", response.json())
# Outcome: status 200 and the title, or status 429 and the message "Too Many
# Requests". Both are normal for the shared pool.

# %% [markdown]
# ## Cell 7.3: The retry schedule
#
# - Every other cell calls `utils.call_api()`, which retries on `429` and `5xx`
#   - It waits 2, 4, 8, then 16 seconds, and then returns the last answer
#   - After the last retry it logs a warning and does not raise, so the notebook
#     keeps running

# %%
# Print the wait schedule and the source of the helper.
print("BACKOFF_SEC=", utils.BACKOFF_SEC)
print(inspect.getsource(utils.call_api))
# Outcome: `BACKOFF_SEC= (2.0, 4.0, 8.0, 16.0)`, then the code of `call_api()`.

# %% [markdown]
# # Part 8: Minimal end to end

# %% [markdown]
# ## Cell 8.1: Search a title and read the citation count
#
# - The flow: search a title, take the best hit, read its citation count, and list
#   its first 3 references
# - Cell 8.1 searches the title and reads the citation count
# - Cell 8.2 lists the first 3 references

# %%
# Search a title, keep the first hit, and read its citation count.
title = "BERT: Pre-training of Deep Bidirectional Transformers"
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/search",
    params={"query": title, "limit": 1, "fields": "title,year,citationCount"},
)
hits = response.json().get("data", [])
best_hit = hits[0] if hits else {}
print("best_hit=", best_hit)
print("citationCount=", best_hit.get("citationCount"))
# Outcome: the best hit is the BERT paper, and `citationCount` is an integer
# larger than 0.

# %% [markdown]
# ## Cell 8.2: List the first 3 references

# %%
# List the first 3 references of the best hit, using its `paperId`.
response = utils.call_api(
    "GET",
    f"{utils.GRAPH_URL}/paper/{best_hit.get('paperId')}/references",
    params={"limit": 3, "fields": "title,year"},
)
first_references = response.json().get("data", [])
display(pd.DataFrame([item["citedPaper"] for item in first_references]))
# Outcome: a table with 3 papers that BERT cites.

# %% [markdown]
# # Part 9: Recommendations (optional)

# %% [markdown]
# ## Primitive 7: `GET /papers/forpaper/{id}`
#
# - **Mental model**: give one paper and get a list of similar papers
#
# | Object | Description | Additional Info |
# |--------|-------------|-----------------|
# | Base URL | `utils.RECOMMENDATIONS_URL` | Not the Academic Graph URL |
# | `GET /papers/forpaper/{id}` | Papers similar to one paper | Same id forms as `/paper/{id}` |
# | `limit` | Number of papers | At most 500 |
# | `recommendedPapers` | List of papers in the answer | Can be empty |

# %% [markdown]
# ## Cell 9.1: Papers similar to BERT

# %%
# Ask for 3 papers similar to BERT.
bert_id = "ARXIV:1810.04805"
response = utils.call_api(
    "GET",
    f"{utils.RECOMMENDATIONS_URL}/papers/forpaper/{bert_id}",
    params={"limit": 3, "fields": "title,year"},
)
recommendations = response.json()
print("keys=", sorted(recommendations))
display(pd.DataFrame(recommendations.get("recommendedPapers", [])))
# Outcome: the key `recommendedPapers` and a table of up to 3 papers. The list can
# be empty for some papers.

# %% [markdown]
# - Official docs of this endpoint:
#   https://api.semanticscholar.org/api-docs/#tag/Paper-Recommendations/operation/get_papers_for_paper

# %% [markdown]
# # Part 10: Summary

# %% [markdown]
# ## Questions to try
#
# - What happens if you remove `fields` from the search in Cell 3.1?
# - What is the largest `limit` that the search accepts?
# - What does the answer look like if you ask for `fields=contexts,isInfluential` on
#   the citations of Cell 6.1?
# - What does `/paper/batch` return if all the ids are unknown?

# %% [markdown]
# ## Summary: The Mental Model
#
# - The API is a graph of `Paper`, `Author`, and `Citation` objects, and a `Venue`
#   is a field of a paper
# - A paper has many ids: use `paperId`, `DOI:<doi>`, or `ARXIV:<id>` in
#   `GET /paper/{id}`
# - You choose the answer with `fields=`, and a list comes as a page with the keys
#   `total`, `offset`, `next`, and `data`
# - Use `POST /paper/batch` to look up many papers in one call
# - A citation item wraps the other paper in the key `citingPaper` or `citedPaper`
# - Expect `429` from the shared pool: wait 2, 4, 8, then 16 seconds, and use a
#   free key with the header `x-api-key` for a limit of your own
# - Citation counts and search results change over time, so check the shape of an
#   answer and not its exact numbers
