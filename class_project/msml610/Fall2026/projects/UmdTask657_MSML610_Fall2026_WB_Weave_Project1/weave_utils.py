"""
weave_utils.py

Utilities for the Weave FOMC tone project. The notebooks call these functions
and classes so the notebook cells stay short and readable.

- `init_weave()` loads the API keys from `.env` and starts a Weave project
- `TfidfLogRegModel`, `FomcRobertaModel`, and `ZeroShotModel` are the three
  hawkish/dovish/neutral classifiers, each written as a `weave.Model`
- `predict_sample()` runs several models on the same sentences for a
  side-by-side comparison
- `make_test_dataset()`, `accuracy_scorer`, `MacroF1Scorer`, and
  `run_evaluations()` evaluate every model on the test set with
  `weave.Evaluation`

Import as:

import weave_utils as wutils
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import pandas as pd
import pydantic
import weave
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent

# Load the API keys (WANDB_API_KEY and HF_TOKEN) from the git-ignored file.
dotenv.load_dotenv(PROJECT_DIR / ".env")

# Cache the Hugging Face models under the git-ignored data/ folder, so they
# survive container restarts and download only once.
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "data" / "hf_cache"))

# Use the label encoding of the Trillion Dollar Words dataset.
LABEL_NAMES: Dict[int, str] = {0: "dovish", 1: "hawkish", 2: "neutral"}
LABEL_IDS: Dict[str, int] = {name: idx for idx, name in LABEL_NAMES.items()}


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Weave setup
# #############################################################################


def init_weave(project_name: str = "msml610-weave-fomc") -> Any:
    """
    Start a Weave project so every `@weave.op` call after this gets traced.

    :param project_name: name of the W&B project that stores the traces
    :return: the Weave client returned by `weave.init()`
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is missing. Add it to the .env file in the project "
            "folder (see https://wandb.ai/authorize)."
        )
    # Turn off the link Weave prints after every traced call, since it floods
    # the notebook when a model predicts hundreds of sentences. Use the
    # environment variable, since passing `settings=` to `weave.init` does not
    # stop the links: a background thread prints them and never sees it.
    os.environ.setdefault("WEAVE_PRINT_CALL_LINK", "false")
    # Show only warnings from the HTTP clients, since at INFO level they print
    # one line per request (`httpx` for W&B, `httpx2` for Hugging Face).
    for http_logger in ("httpx", "httpx2"):
        logging.getLogger(http_logger).setLevel(logging.WARNING)
    client = weave.init(project_name)
    return client


def _format_prediction(label_id: int, confidence: float) -> Dict[str, Any]:
    """
    Put every model's output in the same shape.

    All three models return the numeric label (to compare with the dataset),
    the readable name, and the model's confidence in that label.
    """
    return {
        "label": int(label_id),
        "label_name": LABEL_NAMES[int(label_id)],
        "confidence": round(float(confidence), 4),
    }


# #############################################################################
# Model 1: TF-IDF + logistic regression baseline
# #############################################################################


class TfidfLogRegModel(weave.Model):
    """
    Baseline classifier: TF-IDF features followed by logistic regression.

    The hyperparameters are model attributes, so Weave stores a new version of
    the model whenever one of them changes.
    """

    ngram_range: Tuple[int, int] = (1, 2)
    max_features: int = 10000
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    # Keep the fitted sklearn pipeline private, since Weave only needs to
    # serialize the settings above.
    _pipeline: Optional[Pipeline] = pydantic.PrivateAttr(default=None)

    def fit(self, sentences: List[str], labels: List[int]) -> "TfidfLogRegModel":
        """
        Fit the TF-IDF vectorizer and the classifier on the training split.

        :param sentences: training sentences
        :param labels: integer labels (0 dovish, 1 hawkish, 2 neutral)
        :return: the fitted model, to allow chaining calls
        """
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.ngram_range,
                        lowercase=True,
                        max_features=self.max_features,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self._pipeline.fit(list(sentences), [int(y) for y in labels])
        return self

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() on the training data before predict().")
        probabilities = self._pipeline.predict_proba([sentence])[0]
        classes = self._pipeline.classes_
        best = probabilities.argmax()
        return _format_prediction(classes[best], probabilities[best])


# #############################################################################
# Model 2: pre-trained FOMC-RoBERTa
# #############################################################################


def _label_id_from_hf(raw_label: str) -> int:
    """
    Turn a Hugging Face label such as `LABEL_1` or `Hawkish` into 0, 1, or 2.
    """
    raw_label = raw_label.strip()
    if raw_label.upper().startswith("LABEL_"):
        return int(raw_label.split("_")[-1])
    return LABEL_IDS[raw_label.lower()]


class FomcRobertaModel(weave.Model):
    """
    RoBERTa-large fine-tuned on FOMC sentences by the Trillion Dollar Words
    authors.

    The model uses the same label order as the dataset (LABEL_0 dovish,
    LABEL_1 hawkish, LABEL_2 neutral). The repository is gated on Hugging
    Face, so HF_TOKEN in `.env` has to belong to an account with access.
    """

    model_name: str = "gtfintechlab/FOMC-RoBERTa"
    max_length: int = 256
    _pipe: Any = pydantic.PrivateAttr(default=None)
    # Serialize calls with a lock, since Weave evaluates rows in parallel
    # threads and Hugging Face fast tokenizers fail with "Already borrowed" when
    # two threads use them at the same time.
    _lock: threading.Lock = pydantic.PrivateAttr(default_factory=threading.Lock)

    def _load(self) -> Any:
        # Load the model on the first prediction instead of at construction,
        # so creating the object is instant and the download happens once.
        if self._pipe is None:
            import transformers

            _LOG.info("Loading %s", self.model_name)
            self._pipe = transformers.pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
            )
        return self._pipe

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence with FOMC-RoBERTa.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        with self._lock:
            pipe = self._load()
            output = pipe(sentence, truncation=True, max_length=self.max_length)[0]
        return _format_prediction(_label_id_from_hf(output["label"]), output["score"])


# #############################################################################
# Model 3: zero-shot NLI classifier
# #############################################################################


class ZeroShotModel(weave.Model):
    """
    Zero-shot classifier built on a natural language inference model.

    It never sees our training data. For each sentence, it scores how well
    each label description fits, and the best one wins.
    """

    model_name: str = "facebook/bart-large-mnli"
    # Describe each label in plain words, since the NLI model handles a short
    # explanation better than the bare word "hawkish".
    label_descriptions: Dict[str, str] = {
        "dovish": "dovish, favoring lower interest rates or easier monetary policy",
        "hawkish": "hawkish, favoring higher interest rates or tighter monetary policy",
        "neutral": "neutral, with no clear monetary policy stance",
    }
    hypothesis_template: str = "The monetary policy stance of this sentence is {}."
    _pipe: Any = pydantic.PrivateAttr(default=None)
    # Serialize calls with a lock, since Weave evaluates rows in parallel
    # threads and Hugging Face fast tokenizers fail with "Already borrowed" when
    # two threads use them at the same time.
    _lock: threading.Lock = pydantic.PrivateAttr(default_factory=threading.Lock)

    def _load(self) -> Any:
        if self._pipe is None:
            import transformers

            _LOG.info("Loading %s", self.model_name)
            self._pipe = transformers.pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,
            )
        return self._pipe

    @weave.op
    def predict(self, sentence: str) -> Dict[str, Any]:
        """
        Classify one sentence without any task-specific training.

        :param sentence: FOMC sentence to classify
        :return: dict with `label`, `label_name`, and `confidence`
        """
        descriptions = list(self.label_descriptions.values())
        with self._lock:
            pipe = self._load()
            output = pipe(
                sentence,
                candidate_labels=descriptions,
                hypothesis_template=self.hypothesis_template,
            )
        # Take the first label, since the pipeline sorts the labels by score.
        best_description = output["labels"][0]
        best_name = next(
            name
            for name, text in self.label_descriptions.items()
            if text == best_description
        )
        return _format_prediction(LABEL_IDS[best_name], output["scores"][0])


# #############################################################################
# Side-by-side comparison
# #############################################################################


def predict_sample(
    models: Dict[str, weave.Model],
    sample_df: pd.DataFrame,
    *,
    text_col: str = "sentence",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Run each model on the same sentences and line up the predictions.

    Each `predict()` call is a Weave op, so this also creates one trace per
    model and sentence.

    :param models: display name -> model
    :param sample_df: sentences with their true labels
    :param text_col: column with the sentence text
    :param label_col: column with the true integer label
    :return: one row per sentence, with the true label and each model's label
    """
    rows = []
    for _, row in sample_df.iterrows():
        result = {
            "sentence": row[text_col],
            "true_label": LABEL_NAMES[int(row[label_col])],
        }
        for name, model in models.items():
            prediction = model.predict(row[text_col])
            result[name] = prediction["label_name"]
        rows.append(result)
    return pd.DataFrame(rows)


# #############################################################################
# Evaluation with weave.Evaluation
# #############################################################################


def make_test_dataset(
    test_df: pd.DataFrame,
    *,
    name: str = "fomc-test-sentences",
) -> weave.Dataset:
    """
    Turn the labeled test split into a published `weave.Dataset`.

    The column names have to match the argument names Weave passes around:
    `sentence` goes to each model's `predict()`, and `label` goes to the
    scorers.

    :param test_df: test sentences with `sentence` and `label` columns
    :param name: name of the dataset in Weave
    :return: the published dataset
    """
    rows = [
        {"sentence": str(row.sentence), "label": int(row.label)}
        for row in test_df.itertuples(index=False)
    ]
    dataset = weave.Dataset(name=name, rows=rows)
    weave.publish(dataset)
    return dataset


@weave.op
def accuracy_scorer(label: int, output: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check whether the predicted label matches the true label for one sentence.

    Weave averages `correct` over all rows, which gives the accuracy.
    """
    return {"correct": int(output["label"]) == int(label)}


class MacroF1Scorer(weave.Scorer):
    """
    Macro-F1 over the whole test set.

    F1 can't be computed one sentence at a time, so `score()` only records the
    true and predicted label for each row, and `summarize()` computes macro-F1
    once all the rows are in.
    """

    # Pass the labels explicitly, so a class the model never predicts still
    # counts with an F1 of 0 instead of silently dropping out of the average.
    labels: List[int] = [0, 1, 2]

    @weave.op
    def score(self, *, output: Dict[str, Any], label: int) -> Dict[str, int]:
        return {"true_label": int(label), "predicted_label": int(output["label"])}

    @weave.op
    def summarize(self, score_rows: List[Optional[Dict[str, int]]]) -> Dict[str, Any]:
        rows = [row for row in score_rows if row]
        y_true = [row["true_label"] for row in rows]
        y_pred = [row["predicted_label"] for row in rows]
        per_class = f1_score(
            y_true, y_pred, labels=self.labels, average=None, zero_division=0
        )
        summary = {
            "macro_f1": float(per_class.mean()),
            "n_scored": len(rows),
        }
        for label_id, value in zip(self.labels, per_class):
            summary[f"f1_{LABEL_NAMES[label_id]}"] = float(value)
        return summary


def make_evaluation(dataset: weave.Dataset) -> weave.Evaluation:
    """
    Build the evaluation used for every tone model.

    :param dataset: the published test dataset
    :return: an evaluation with the accuracy and macro-F1 scorers
    """
    return weave.Evaluation(
        name="fomc-tone-evaluation",
        dataset=dataset,
        scorers=[accuracy_scorer, MacroF1Scorer()],
    )


async def run_evaluations(
    evaluation: weave.Evaluation,
    models: Dict[str, weave.Model],
) -> pd.DataFrame:
    """
    Evaluate each model and collect the headline numbers in one table.

    Make one warm-up prediction before each evaluation. If a model can't
    load (for example, a gated Hugging Face repo without access), the error
    shows up here, instead of Weave recording a failed call for every row and
    still finishing the evaluation. The warm-up also keeps the one-time model
    loading out of the latency numbers.

    :param evaluation: evaluation from `make_evaluation()`
    :param models: display name -> model
    :return: one row per model with macro-F1, accuracy, and latency
    """
    first_sentence = evaluation.dataset.rows[0]["sentence"]
    n_rows = len(evaluation.dataset.rows)
    # Score one row at a time. With the default of 20 parallel threads, the
    # other threads wait on the lock of the transformer models, and Weave counts
    # that wait as model latency (it made the zero-shot model look ten times
    # slower). One row at a time gives honest latencies at the same total
    # speed.
    os.environ["WEAVE_PARALLELISM"] = "1"
    # Keep only warnings from Weave while the evaluations run, since it logs a
    # progress line per row (about 1,500 lines for three models) and this
    # function returns a summary table anyway.
    weave_logger = logging.getLogger("weave")
    previous_level = weave_logger.level
    weave_logger.setLevel(logging.WARNING)
    results = []
    try:
        for name, model in models.items():
            _LOG.info("Evaluating %s on %d sentences", name, n_rows)
            model.predict(first_sentence)
            summary = await evaluation.evaluate(
                model, __weave={"display_name": name}
            )
            results.append(_summary_row(name, summary, n_rows))
    finally:
        weave_logger.setLevel(previous_level)
    return pd.DataFrame(results).set_index("model").round(4)


def _summary_row(name: str, summary: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
    """
    Pick out the reported numbers from one `evaluate()` summary.
    """
    f1_summary = summary["MacroF1Scorer"]
    if f1_summary["n_scored"] < n_rows:
        _LOG.warning(
            "%s: only %d of %d rows were scored; check the failed calls in Weave",
            name,
            f1_summary["n_scored"],
            n_rows,
        )
    return {
        "model": name,
        "macro_f1": f1_summary["macro_f1"],
        "accuracy": summary["accuracy_scorer"]["correct"]["true_fraction"],
        "f1_dovish": f1_summary["f1_dovish"],
        "f1_hawkish": f1_summary["f1_hawkish"],
        "f1_neutral": f1_summary["f1_neutral"],
        "mean_latency_s": summary["model_latency"]["mean"],
    }
