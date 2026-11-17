"""
Utilities for the W&B Weave tutorial.

Import as:

import tutorials.Weave.weave_utils as utils
"""

import asyncio
import contextvars
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import dotenv
import pandas as pd
import weave

import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebo
import inspect
import helpers.hintrospection as hintros

_LOG = logging.getLogger(__name__)


# #############################################################################
# Notebook configuration
# #############################################################################


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utility logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


def configure_weave() -> None:
    """
    Load W&B credentials and configure tracing output.
    """
    dotenv.load_dotenv()
    hdbg.dassert(
        bool(os.environ.get("WANDB_API_KEY")),
        "Set WANDB_API_KEY before running the tutorial.",
    )
    os.environ["WEAVE_PRINT_CALL_LINK"] = "false"
    for logger_name in ("httpx", "httpx2"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def print_api_info(obj: Any) -> None:
    """
    Display the public interface of an installed API object.

    Unlike print_obj_info(), this function does not require
    the object's source files to belong to a Git repository.

    :param obj: function, method, class, or instance to inspect
    """
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        info = hintros.get_function_info_as_str(
            obj,
            use_markdown=True,
        )
    else:
        info = hintros.get_public_methods_as_str(
            obj,
            use_markdown=True,
        )
    print(info)


# #############################################################################
# Keyword classification
# #############################################################################


@weave.op
def count_keyword_hits(sentence: str, words: List[str]) -> int:
    """
    Count keywords that match the beginning of a token.

    :param sentence: text to search
    :param words: keywords to match
    :return: number of matching keywords
    """
    tokens = re.findall(r"[a-z]+", sentence.lower())
    return sum(
        any(token.startswith(word) for token in tokens)
        for word in words
    )


@weave.op
def keyword_tone(
    sentence: str,
    hawkish_words: List[str],
    dovish_words: List[str],
) -> Dict[str, Any]:
    """
    Classify a sentence using monetary policy keywords.

    :param sentence: text to classify
    :param hawkish_words: keywords associated with tighter policy
    :param dovish_words: keywords associated with easier policy
    :return: predicted label and keyword counts
    """
    hawkish_hits = count_keyword_hits(sentence, hawkish_words)
    dovish_hits = count_keyword_hits(sentence, dovish_words)
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


# #############################################################################
# Versioned models
# #############################################################################


class KeywordToneModel(weave.Model):
    """
    Classify monetary policy tone using configurable keyword lists.
    """

    hawkish_words: List[str]
    dovish_words: List[str]

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Predict the monetary policy tone of a sentence.

        :param sentence: text to classify
        :return: predicted label and keyword counts
        """
        return keyword_tone(
            sentence,
            self.hawkish_words,
            self.dovish_words,
        )


# #############################################################################
# Evaluation scorers
# #############################################################################


@weave.op
def exact_match(
    label: str,
    output: Dict[str, Any],
) -> Dict[str, bool]:
    """
    Check whether a predicted label matches its reference label.

    :param label: reference label
    :param output: model prediction
    :return: dictionary containing prediction correctness
    """
    return {"correct": output["label"] == label}


class PerClassRecall(weave.Scorer):
    """
    Calculate recall separately for each sentiment class.
    """

    @weave.op
    def score(
        self,
        *,
        output: Dict[str, Any],
        label: str,
    ) -> Dict[str, Any]:
        """
        Record prediction correctness and the reference class.

        :param output: model prediction
        :param label: reference label
        :return: reference label and prediction correctness
        """
        return {
            "label": label,
            "correct": output["label"] == label,
        }

    @weave.op
    def summarize(
        self,
        score_rows: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Aggregate prediction correctness by reference class.

        :param score_rows: individual scorer outputs
        :return: recall for each observed class
        """
        rows = [row for row in score_rows if row]
        if not rows:
            return {}
        scores = pd.DataFrame(rows)
        recall = scores.groupby("label")["correct"].mean()
        return {
            f"recall_{label}": float(value)
            for label, value in recall.items()
        }


# #############################################################################
# Evaluation execution and reporting
# #############################################################################


def run_evaluation(
    evaluation: weave.Evaluation,
    model: weave.Model,
    display_name: str,
) -> Dict[str, Any]:
    """
    Run an evaluation from either a notebook or a Python script.

    :param evaluation: configured Weave evaluation
    :param model: model to evaluate
    :param display_name: name displayed in the Weave interface
    :return: evaluation summary
    """
    async def evaluate() -> Dict[str, Any]:
        return await evaluation.evaluate(
            model,
            __weave={"display_name": display_name},
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(evaluate())

    # Jupyter already has a running event loop.
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            context.run,
            lambda: asyncio.run(evaluate()),
        )
        return future.result()


def summarize_keyword_evaluations(
    summaries: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert keyword model evaluation summaries into a comparison table.

    :param summaries: evaluation summaries keyed by model version
    :return: comparison table with one row per model
    """
    rows = []
    for version, summary in summaries.items():
        rows.append(
            {
                "model": version,
                "accuracy": summary["exact_match"]["correct"][
                    "true_fraction"
                ],
                **summary["PerClassRecall"],
                "mean_latency_s": summary["model_latency"]["mean"],
            }
        )
    return pd.DataFrame(rows).set_index("model").round(3)