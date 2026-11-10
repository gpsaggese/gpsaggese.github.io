"""
Utility functions for the Semantic Scholar API tutorial notebook.

- The notebook calls these functions instead of writing the throttle, the retry,
  and the diagram code inline
- The API calls are plain `requests` calls: no `semanticscholar` client is used
- The API key is read only from the environment variable
  `SEMANTIC_SCHOLAR_API_KEY`, sent only in the header `x-api-key`, and never
  printed or logged

Import as:

import semantic_scholar_utils as utils
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import graphviz
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import requests

_LOG = logging.getLogger(__name__)


# #############################################################################
# Constants
# #############################################################################

# Base URLs of the two services that the notebook calls.
GRAPH_URL = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_URL = "https://api.semanticscholar.org/recommendations/v1"

# Seconds to wait after each request: a key allows 1 request per second.
THROTTLE_SEC = 1.0

# Seconds to wait before each retry of a `429` or `5xx` answer.
BACKOFF_SEC = (2.0, 4.0, 8.0, 16.0)

# Name of the environment variable that holds the API key.
_API_KEY_ENV_VAR = "SEMANTIC_SCHOLAR_API_KEY"

# Seconds to wait for one answer before the request fails.
_TIMEOUT_SEC = 30.0


# #############################################################################
# Setup
# #############################################################################


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# API calls
# #############################################################################


def _is_retryable(status_code: int) -> bool:
    """
    Check if an answer means "try again later".

    :param status_code: HTTP status of the answer
    :return: True for `429` (rate limit) and for every `5xx` (server error)
    """
    is_retryable = status_code == 429 or 500 <= status_code < 600
    return is_retryable


def _get_headers(*, use_key: bool) -> Dict[str, str]:
    """
    Build the headers of a request.

    The key goes only in the header `x-api-key`. The key is not logged.

    :param use_key: send the key if the environment holds one
    :return: headers with `x-api-key`, or an empty dict without a key
    """
    headers: Dict[str, str] = {}
    api_key = os.environ.get(_API_KEY_ENV_VAR)
    if use_key and api_key:
        headers["x-api-key"] = api_key
    return headers


def _send(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]],
    json_body: Optional[Dict[str, Any]],
    headers: Dict[str, str],
) -> requests.Response:
    """
    Send one request, log its status, and wait to stay below the rate limit.

    :param method: HTTP method, `GET` or `POST`
    :param url: full URL of the endpoint
    :param params: query parameters
    :param json_body: JSON body of a `POST` request
    :param headers: headers of the request
    :return: the answer, also when its status is not `200`
    """
    response = requests.request(
        method,
        url,
        params=params,
        json=json_body,
        headers=headers,
        timeout=_TIMEOUT_SEC,
    )
    _LOG.info("%s %s -> status %s", method, url, response.status_code)
    # Stay below 1 request per second.
    time.sleep(THROTTLE_SEC)
    return response


def call_api(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    use_key: bool = True,
    backoff_sec: Tuple[float, ...] = BACKOFF_SEC,
) -> requests.Response:
    """
    Call the API with a throttle and a retry on `429` and `5xx`.

    The function waits `THROTTLE_SEC` after each request. On `429` or `5xx` it
    waits for each value of `backoff_sec` (2, 4, 8, then 16 seconds) and sends
    the request again. After the last retry it returns the last answer and
    logs a warning: it never raises on a `429`, so the notebook keeps running.

    :param method: HTTP method, `GET` or `POST`
    :param url: full URL of the endpoint
    :param params: query parameters, e.g., `{"fields": "title,year"}`
    :param json_body: JSON body of a `POST` request
    :param use_key: send the API key if the environment holds one
        - Pass False to see the answer of the shared pool without a key
    :param backoff_sec: seconds to wait before each retry
        - Pass `()` to send only one request
    :return: the last answer, also when its status is not `200`
    """
    hdbg.dassert_in(method, ["GET", "POST"])
    headers = _get_headers(use_key=use_key)
    response = _send(
        method, url, params=params, json_body=json_body, headers=headers
    )
    # Retry once for each backoff value, until the answer is not retryable.
    for attempt, wait_sec in enumerate(backoff_sec, start=1):
        if not _is_retryable(response.status_code):
            break
        _LOG.info("Wait %s s before retry %s", wait_sec, attempt)
        time.sleep(wait_sec)
        response = _send(
            method, url, params=params, json_body=json_body, headers=headers
        )
    if _is_retryable(response.status_code):
        _LOG.warning(
            "Status %s after %s retries: the rate limit is hit. Without a key "
            "all users share one pool, so set the environment variable '%s' "
            "to get a limit of your own",
            response.status_code,
            len(backoff_sec),
            _API_KEY_ENV_VAR,
        )
    return response


# #############################################################################
# Cell 1.1: The base URL and the API key
# #############################################################################


def cell1_1_has_api_key() -> bool:
    """
    Check if the environment holds an API key, without reading it out.

    :return: True if `SEMANTIC_SCHOLAR_API_KEY` is set and not empty
    """
    has_key = bool(os.environ.get(_API_KEY_ENV_VAR))
    return has_key


# #############################################################################
# Cell 1.2: The objects of the graph
# #############################################################################


def cell1_2_get_object_diagram() -> graphviz.Source:
    """
    Build the diagram of `Paper`, `Author`, `Citation`, and `Venue`.

    A `Citation` is an edge between 2 papers, drawn as a node to show its 2
    wrapper keys `citingPaper` and `citedPaper`. A `Venue` is a field of a
    `Paper`: the API has no endpoint for venues.

    :return: Graphviz source, rendered by the notebook
    """
    # author : green | paper : blue | citation : orange | venue : purple
    dot = """
    digraph SemanticScholar {
      rankdir=LR;
      bgcolor="transparent";
      nodesep=0.5;
      ranksep=0.9;
      node [shape=box, style="rounded,filled", fontname="Helvetica"];
      edge [fontname="Helvetica", fontsize=10, color="#A3B1C0",
            fontcolor="#7B8794"];
      Author [label=<<b>Author</b><br/>authorId, name, hIndex>,
              fillcolor="#A9DDB0", color="#4F9A5C", fontcolor="#1F4E2E"];
      Paper [label=<<b>Paper</b><br/>paperId, title, year, citationCount>,
             fillcolor="#9CC4F2", color="#3C6FB0", fontcolor="#1F4E79"];
      Venue [label=<<b>Venue</b><br/>venue, publicationVenue<br/>(fields of Paper)>,
             fillcolor="#E8D9F7", color="#9B7DB1", fontcolor="#4A2E5C"];
      Citation [label=<<b>Citation</b><br/>contexts, isInfluential>,
                fillcolor="#FFC98A", color="#D98E2B", fontcolor="#6B4517"];
      Author -> Paper [label="authors / papers", dir=both];
      Paper -> Venue [label="venue"];
      Citation -> Paper [label="citingPaper"];
      Citation -> Paper [label="citedPaper"];
    }
    """
    diagram = graphviz.Source(dot)
    return diagram
