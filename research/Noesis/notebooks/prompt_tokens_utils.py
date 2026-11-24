"""
Fit and plot the prompt tokens the seed sellers bill per character (#720).

The calls come from `scripts/experiment_prompt_tokens.py`; this module turns
them into the tables and plots of `notebooks/prompt_tokens.ipynb`.

Import as:

import research.Noesis.notebooks.prompt_tokens_utils as rnnoprtout
"""

import collections
import dataclasses
import logging
import math
import statistics
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import helpers.hnotebook as hnotebo

import research.Noesis.gateway.gateway_providers as rnogapro
import research.Noesis.scripts.experiment_prompt_tokens as rnsexprto

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


@dataclasses.dataclass(frozen=True)
class Fit:
    """
    `prompt_tokens = n_chars / chars_per_token + request_overhead`, plus
    `message_overhead` per extra message.
    """

    chars_per_token: float
    request_overhead: float
    message_overhead: float
    # Calls behind each fitted number.
    n_single: int
    n_split: int


def fit(
    measurements: List[rnsexprto.Measurement],
) -> Dict[Tuple[str, str], Fit]:
    """
    Fit chars per token and the two overheads per (seller, text type).

    :param measurements: the calls of one run
    :return: one fit per (seller, text type) with at least two one-message
        lengths
    """
    groups = collections.defaultdict(list)
    for m in measurements:
        if m.status == rnogapro.OK and m.prompt_tokens is not None:
            groups[(m.seller_id, m.text_type)].append(m)
    fits = {}
    for key, ms in sorted(groups.items()):
        # Fit a line on the one-message calls; skip groups with too few lengths.
        single = {m.n_chars: m.prompt_tokens for m in ms if m.n_messages == 1}
        if len(single) < 2:
            continue
        slope, intercept = statistics.linear_regression(
            list(single), list(single.values())
        )
        # Each split call costs this much more than the same text as one message.
        extra = [
            (m.prompt_tokens - single[m.n_chars]) / (m.n_messages - 1)
            for m in ms
            if m.n_messages > 1 and m.n_chars in single
        ]
        fits[key] = Fit(
            chars_per_token=1 / slope if slope > 0 else math.inf,
            request_overhead=intercept,
            message_overhead=statistics.fmean(extra) if extra else math.nan,
            n_single=len(single),
            n_split=len(extra),
        )
    return fits


# #############################################################################
# Cell 1.1: Load the Raw Calls
# #############################################################################


def cell1_1_summarize_calls(
    measurements: List[rnsexprto.Measurement],
) -> pd.DataFrame:
    """
    Count the calls, the successes, and the spend of each seller.

    :param measurements: the calls of one run
    :return: one row per seller, with the statuses of the failed calls
    """
    rows = []
    sellers = sorted({m.seller_id for m in measurements})
    for seller_id in sellers:
        ms = [m for m in measurements if m.seller_id == seller_id]
        failed = collections.Counter(
            m.status for m in ms if m.status != rnogapro.OK
        )
        rows.append(
            {
                "seller": seller_id,
                "calls": len(ms),
                "succeeded": sum(m.status == rnogapro.OK for m in ms),
                "failed": ", ".join(
                    f"{n} x {status}" for status, n in sorted(failed.items())
                ),
                "spend_usd": round(sum(m.cost_usd for m in ms), 4),
            }
        )
    df = pd.DataFrame(rows)
    return df


# #############################################################################
# Cell 1.2: Fit Characters per Token and Framing
# #############################################################################


def cell1_2_fit_table(fits: Dict[Tuple[str, str], Fit]) -> pd.DataFrame:
    """
    Show one fit per seller and text type.

    :param fits: the output of `fit()`
    :return: one row per (seller, text type), rounded as in the research doc
    """
    rows = [
        {
            "seller": seller_id,
            "text": text_type,
            "chars_per_token": round(f.chars_per_token, 2),
            "request_overhead": round(f.request_overhead, 1),
            "per_message_overhead": round(f.message_overhead, 1),
            "one_message_calls": f.n_single,
            "split_calls": f.n_split,
        }
        for (seller_id, text_type), f in fits.items()
    ]
    df = pd.DataFrame(rows)
    return df


# #############################################################################
# Cell 2.1: Billed Tokens against Characters
# #############################################################################


def cell2_1_plot_tokens_vs_chars(
    measurements: List[rnsexprto.Measurement],
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the billed prompt tokens of the one-message calls, one panel per text
    type and one line per seller.

    :param measurements: the calls of one run
    :param figsize: figure size; defaults to `plt.rcParams["figure.figsize"]`
    """
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    ok = [
        m for m in measurements if m.status == rnogapro.OK and m.n_messages == 1
    ]
    text_types = sorted({m.text_type for m in ok})
    # Keep `axes` a list even with a single text type.
    _, axes = plt.subplots(
        1, len(text_types), figsize=figsize, sharey=True, squeeze=False
    )
    axes = axes[0]
    for ax, text_type in zip(axes, text_types):
        for seller_id in sorted({m.seller_id for m in ok}):
            ms = sorted(
                (
                    m
                    for m in ok
                    if m.seller_id == seller_id and m.text_type == text_type
                ),
                key=lambda m: m.n_chars,
            )
            ax.plot(
                [m.n_chars for m in ms],
                [m.prompt_tokens for m in ms],
                marker="o",
                label=seller_id,
            )
        ax.set_title(text_type)
        ax.set_xlabel("characters")
    axes[0].set_ylabel("billed prompt tokens")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
