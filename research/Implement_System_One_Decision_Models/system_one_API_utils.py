"""Small client and displays for the System One decision API notebook.

Import as:

import research.Implement_System_One_Decision_Models.system_one_API_utils as soau
"""

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

# #############################################################################
# API client
# #############################################################################


DEFAULT_URL = "http://127.0.0.1:18089/v1/systemone"


class SystemOneClient:
    """Call llama.cpp's typed decision endpoint with one state per request."""

    def __init__(self, url: Optional[str] = None) -> None:
        """Use an explicit URL or the `SYSTEM_ONE_URL` environment variable."""
        self.url = url or os.environ.get("SYSTEM_ONE_URL", DEFAULT_URL)

    def system_one(
        self, state: Any, questions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return typed answers for the named questions over one state."""
        payload = {"state": state, "questions": questions}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as connection:
                result = json.load(connection)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Decision API returned HTTP {error.code}: {detail}"
            ) from error
        if not isinstance(result, dict) or set(result.get("answers", {})) != set(
            questions
        ):
            raise ValueError("Response does not contain the requested answers")
        return result


# #############################################################################
# Answer interpretation
# #############################################################################


def choice_probabilities(
    answer: Dict[str, Any], option_names: List[str]
) -> Dict[str, float]:
    """Return a choice distribution with its declared option names intact."""
    probabilities = answer["probabilities"]
    if answer.get("type") != "choice" or set(probabilities) != set(option_names):
        raise ValueError(
            "Choice probabilities do not match the declared options"
        )
    values = {name: float(probabilities[name]) for name in option_names}
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in values.values()
    ):
        raise ValueError("Choice probabilities must be finite and within [0, 1]")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-3):
        raise ValueError("Choice probabilities must sum to one")
    return values


def choose_top_option(probabilities: Dict[str, float]) -> str:
    """Return the option with highest probability, with stable tie order."""
    if not probabilities:
        raise ValueError("The choice distribution is empty")
    return max(probabilities, key=probabilities.get)


def plot_probabilities(
    probabilities: Dict[str, float], title: str
) -> plt.Figure:
    """Plot a small categorical distribution without changing its values."""
    labels = list(probabilities)
    values = [probabilities[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(labels) * 0.6)))
    ax.barh(labels, values, color="#3483c7")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model probability")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
