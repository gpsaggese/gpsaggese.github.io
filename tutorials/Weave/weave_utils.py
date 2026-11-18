"""
Utilities for the W&B Weave tutorial.

Import as:

import tutorials.Weave.weave_utils as utils
"""

import asyncio
import contextvars
import hashlib
import inspect
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import dotenv
import helpers.hdbg as hdbg
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import requests
import seaborn as sns
import weave
from bs4 import BeautifulSoup
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

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
def count_keyword_hits(sentence: str, words: list[str]) -> int:
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
    hawkish_words: list[str],
    dovish_words: list[str],
) -> dict[str, Any]:
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

    hawkish_words: list[str]
    dovish_words: list[str]

    @weave.op
    def predict(self, sentence: str) -> dict[str, Any]:
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
    output: dict[str, Any],
) -> dict[str, bool]:
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
        output: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
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
        score_rows: list[dict[str, Any]],
    ) -> dict[str, float]:
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
) -> dict[str, Any]:
    """
    Run an evaluation from either a notebook or a Python script.

    :param evaluation: configured Weave evaluation
    :param model: model to evaluate
    :param display_name: name displayed in the Weave interface
    :return: evaluation summary
    """
    async def evaluate() -> dict[str, Any]:
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
    summaries: dict[str, dict[str, Any]],
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


# #############################################################################
# FOMC application data
# #############################################################################


DATA_DIR = Path(__file__).resolve().parent / "data"

FOMC_DATASET_URL = (
    "https://huggingface.co/datasets/"
    "gtfintechlab/fomc_communication/resolve/main"
)

LABEL_NAMES = {
    0: "dovish",
    1: "hawkish",
    2: "neutral",
}


def load_fomc_split(split: str) -> pd.DataFrame:
    """
    Download, cache, and validate a labeled FOMC dataset split.

    :param split: dataset split, either train or test
    :return: cleaned dataframe containing sentences and labels
    """
    hdbg.dassert_in(split, ["train", "test"])
    cache_dir = DATA_DIR / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}.csv"

    if not cache_path.exists():
        url = f"{FOMC_DATASET_URL}/{split}.csv"
        _LOG.info("Downloading %s", url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)

    data = pd.read_csv(cache_path)
    data = data[["sentence", "year", "label"]].copy()

    hdbg.dassert(data["sentence"].notna().all())
    hdbg.dassert(data["label"].notna().all())

    data["sentence"] = data["sentence"].str.strip()

    hdbg.dassert(data["sentence"].ne("").all())
    hdbg.dassert(
        set(data["label"].unique()).issubset(LABEL_NAMES)
    )

    data["label"] = data["label"].astype(int)
    return data


def training_data_digest(train_df: pd.DataFrame) -> str:
    """
    Calculate a reproducible fingerprint of the training data.

    :param train_df: training sentences and labels
    :return: SHA-256 fingerprint of the training examples
    """
    content = train_df[["sentence", "label"]].to_csv(
        index=False,
        lineterminator="\n",
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# #############################################################################
# FOMC classification models
# #############################################################################


class TfidfLogRegModel(weave.Model):
    """
    Classify FOMC sentences using TF-IDF and logistic regression.

    Weave versions the model configuration and training data fingerprint.
    The fitted scikit-learn pipeline is kept in memory.
    """

    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 10000
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    class_weight: str | None = None
    training_digest: str

    _pipeline: Pipeline | None = pydantic.PrivateAttr(
        default=None
    )

    def fit(
        self,
        train_df: pd.DataFrame,
    ) -> "TfidfLogRegModel":
        """
        Train the model on labeled FOMC sentences.

        :param train_df: training sentences and labels
        :return: fitted model
        """
        hdbg.dassert_eq(
            self.training_digest,
            training_data_digest(train_df),
        )

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
                        class_weight=self.class_weight,
                    ),
                ),
            ]
        )

        self._pipeline.fit(
            train_df["sentence"].tolist(),
            train_df["label"].tolist(),
        )
        return self

    @weave.op
    def predict(self, sentence: str) -> dict[str, Any]:
        """
        Classify a single FOMC sentence.

        :param sentence: sentence to classify
        :return: predicted label, label name, and confidence
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Call fit() before predict()."
            )

        probabilities = self._pipeline.predict_proba(
            [sentence]
        )[0]

        best = probabilities.argmax()
        label = int(self._pipeline.classes_[best])

        return {
            "label": label,
            "label_name": LABEL_NAMES[label],
            "confidence": round(
                float(probabilities[best]), 4
            ),
        }

    @weave.op
    def predict_batch(
        self,
        sentences: list[str],
    ) -> list[int]:
        """
        Classify multiple FOMC sentences in one operation.

        :param sentences: monetary policy sentences
        :return: predicted numeric labels
        """
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict().")

        hdbg.dassert(len(sentences) > 0)

        predictions = self._pipeline.predict(sentences)

        return [int(label) for label in predictions]



# #############################################################################
# FOMC model evaluation
# #############################################################################


def make_fomc_test_dataset(
    test_df: pd.DataFrame,
) -> weave.Dataset:
    """
    Convert labeled test sentences into a Weave dataset.

    :param test_df: held-out sentences and labels
    :return: dataset for model evaluation
    """
    rows = [
        {
            "sentence": str(row.sentence),
            "label": int(row.label),
        }
        for row in test_df.itertuples(index=False)
    ]

    return weave.Dataset(
        name="fomc-test-sentences",
        rows=rows,
    )


@weave.op
def accuracy_scorer(
    label: int,
    output: dict[str, Any],
) -> dict[str, bool]:
    """
    Check whether a prediction matches the reference label.

    :param label: true numeric label
    :param output: model prediction
    :return: dictionary indicating prediction correctness
    """
    return {
        "correct": int(output["label"]) == int(label)
    }


class MacroF1Scorer(weave.Scorer):
    """
    Calculate macro-F1 across the three monetary policy classes.
    """

    labels: list[int] = pydantic.Field(
        default_factory=lambda: [0, 1, 2],
    )

    @weave.op
    def score(
        self,
        *,
        output: dict[str, Any],
        label: int,
    ) -> dict[str, int]:
        """
        Record the true and predicted labels.

        :param output: model prediction
        :param label: true numeric label
        :return: labels needed for aggregate scoring
        """
        return {
            "true_label": int(label),
            "predicted_label": int(output["label"]),
        }

    @weave.op
    def summarize(
        self,
        score_rows: list[dict[str, int] | None],
    ) -> dict[str, Any]:
        """
        Calculate macro-F1 and individual class F1 scores.

        :param score_rows: results from individual predictions
        :return: aggregate classification metrics
        """
        rows = [row for row in score_rows if row]
        hdbg.dassert(len(rows) > 0)

        y_true = [row["true_label"] for row in rows]
        y_pred = [row["predicted_label"] for row in rows]

        per_class = f1_score(
            y_true,
            y_pred,
            labels=self.labels,
            average=None,
            zero_division=0,
        )

        summary = {
            "macro_f1": float(per_class.mean()),
            "n_scored": len(rows),
        }

        for label_id, value in zip(self.labels, per_class):
            summary[f"f1_{LABEL_NAMES[label_id]}"] = float(value)

        return summary


def summarize_fomc_evaluations(
    summaries: dict[str, dict[str, Any]],
    expected_rows: int,
) -> pd.DataFrame:
    """
    Compare the classification metrics of evaluated models.

    :param summaries: model names and evaluation summaries
    :param expected_rows: expected number of scored sentences
    :return: model comparison dataframe
    """
    rows = []

    for name, summary in summaries.items():
        f1_summary = summary["MacroF1Scorer"]

        hdbg.dassert_eq(
            f1_summary["n_scored"],
            expected_rows,
        )

        rows.append(
            {
                "model": name,
                "macro_f1": f1_summary["macro_f1"],
                "accuracy": summary["accuracy_scorer"][
                    "correct"
                ]["true_fraction"],
                "f1_dovish": f1_summary["f1_dovish"],
                "f1_hawkish": f1_summary["f1_hawkish"],
                "f1_neutral": f1_summary["f1_neutral"],
                "mean_latency_s": summary["model_latency"][
                    "mean"
                ],
                "n_scored": f1_summary["n_scored"],
            }
        )

    return pd.DataFrame(rows).set_index("model")


# #############################################################################
# FOMC statement ingestion
# #############################################################################


FED_BASE_URL = "https://www.federalreserve.gov"


def discover_fomc_statements(
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Discover official FOMC monetary policy statement URLs.

    :param start_year: first announcement year
    :param end_year: last announcement year
    :return: announcement dates and source URLs
    """
    calendar_urls = [
        f"{FED_BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
        for year in range(start_year, min(end_year, 2020) + 1)
    ]

    calendar_urls.append(
        f"{FED_BASE_URL}/monetarypolicy/fomccalendars.htm"
    )

    pattern = re.compile(
        r"/newsevents/pressreleases/monetary(\d{8})a\.htm",
        re.IGNORECASE,
    )

    links = {}

    for calendar_url in calendar_urls:
        response = requests.get(calendar_url, timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for link in soup.find_all("a", href=True):
            url = urljoin(calendar_url, link["href"])
            match = pattern.search(url)

            if match:
                date = pd.to_datetime(
                    match.group(1),
                    format="%Y%m%d",
                )

                if start_year <= date.year <= end_year:
                    links[date] = url

    hdbg.dassert(len(links) > 0)

    # Exclude the unrelated FIMA Repo Facility announcement.
    links.pop(pd.Timestamp("2020-03-31"), None)

    return (
        pd.DataFrame(
            [
                {"date": date, "source_url": url}
                for date, url in links.items()
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def download_fomc_statement(
    date: pd.Timestamp,
    url: str,
) -> str:
    """
    Download and cache an official FOMC statement.

    :param date: announcement date
    :param url: official statement URL
    :return: cleaned statement text
    """
    cache_dir = DATA_DIR / "raw" / "fomc_statements"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{date:%Y-%m-%d}.txt"

    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if len(text.strip()) >= 300:
            return text

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    article = soup.select_one("#article")

    hdbg.dassert(
        article is not None,
        f"No article found for {date}",
    )

    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in article.find_all("p")
    ]

    text = "\n".join(
        paragraph for paragraph in paragraphs if paragraph
    )

    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text).strip()

    hdbg.dassert(
        len(text) >= 300,
        f"Statement appears too short: {date}",
    )

    cache_path.write_text(text, encoding="utf-8")

    time.sleep(0.3)
    return text


def load_fomc_statements(
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Load and cache official FOMC monetary policy statements.

    :param start_year: first announcement year
    :param end_year: last announcement year
    :return: dates, statement texts, and source URLs
    """
    cache_dir = DATA_DIR / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        cache_dir
        / f"fomc_statements_{start_year}_{end_year}.csv"
    )

    if cache_path.exists():
        data = pd.read_csv(
            cache_path,
            parse_dates=["date"],
        )
    else:
        links = discover_fomc_statements(
            start_year,
            end_year,
        )

        records = []

        for row in links.itertuples(index=False):
            text = download_fomc_statement(
                row.date,
                row.source_url,
            )

            records.append(
                {
                    "date": row.date,
                    "statement_text": text,
                    "source_url": row.source_url,
                }
            )

        data = pd.DataFrame(records)

    data = data.sort_values("date").reset_index(drop=True)

    hdbg.dassert(data["date"].is_unique)
    hdbg.dassert(data["statement_text"].notna().all())
    hdbg.dassert(data["statement_text"].str.strip().ne("").all())

    if (start_year, end_year) == (2015, 2024):
        hdbg.dassert_eq(len(data), 84)

    data.to_csv(cache_path, index=False)
    return data


# #############################################################################
# Treasury yield ingestion
# #############################################################################


def download_treasury_year(
    year: int,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Download and cache daily Treasury yields for one year.

    :param year: observation year
    :param max_retries: maximum download attempts
    :return: raw Treasury yield dataframe
    """
    hdbg.dassert(max_retries > 0)

    cache_dir = DATA_DIR / "raw" / "treasury_yields"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"treasury_{year}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = (
        "https://home.treasury.gov/resource-center/"
        "data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
    )

    params = {
        "type": "daily_treasury_yield_curve",
        "field_tdr_date_value": str(year),
        "page": "",
        "_format": "csv",
    }

    for attempt in range(max_retries):
        try:
            _LOG.info("Downloading Treasury yields for %d", year)

            response = requests.get(
                url,
                params=params,
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            data = pd.read_csv(StringIO(response.text))
            data.columns = data.columns.str.strip()

            hdbg.dassert(not data.empty)
            hdbg.dassert_in("Date", data.columns)
            hdbg.dassert_in("2 Yr", data.columns)

            data.to_csv(cache_path, index=False)

            time.sleep(1)
            return data

        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise

            wait_seconds = min(5 * 2**attempt, 60)

            _LOG.warning(
                "Treasury download failed; retrying in %d seconds",
                wait_seconds,
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Treasury download failed.")


def load_treasury_yields(
    start_year: int = 2014,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Load daily 2-year Treasury par yields.

    :param start_year: first observation year
    :param end_year: last observation year
    :return: sorted dates and 2-year yields in percent
    """
    hdbg.dassert(start_year <= end_year)

    cache_dir = DATA_DIR / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        cache_dir
        / f"treasury_2yr_daily_{start_year}_{end_year}.csv"
    )

    if cache_path.exists():
        data = pd.read_csv(
            cache_path,
            parse_dates=["date"],
        )
    else:
        frames = [
            download_treasury_year(year)
            for year in range(start_year, end_year + 1)
        ]

        raw = pd.concat(frames, ignore_index=True)
        raw.columns = raw.columns.str.strip()

        data = raw[["Date", "2 Yr"]].rename(
            columns={
                "Date": "date",
                "2 Yr": "yield_2yr",
            }
        )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        )

        data["yield_2yr"] = pd.to_numeric(
            data["yield_2yr"],
            errors="coerce",
        )

        data = data.dropna(
            subset=["date", "yield_2yr"]
        )

    data = data.sort_values("date").reset_index(drop=True)

    hdbg.dassert(not data.empty)
    hdbg.dassert(data["date"].is_unique)
    hdbg.dassert(data["yield_2yr"].notna().all())
    hdbg.dassert(
        data["date"].dt.year.between(
            start_year,
            end_year,
        ).all()
    )

    data.to_csv(cache_path, index=False)
    return data


# #############################################################################
# FOMC and Treasury alignment
# #############################################################################


def align_fomc_yields(
    statements: pd.DataFrame,
    treasury: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match announcements to same-day or next-trading-day yields.

    :param statements: dated FOMC monetary policy statements
    :param treasury: daily 2-year Treasury yields
    :return: statements with aligned yield changes in basis points
    """
    statements = statements.copy()
    treasury = treasury.copy()

    statements["date"] = pd.to_datetime(
        statements["date"]
    ).dt.normalize()

    treasury["date"] = pd.to_datetime(
        treasury["date"]
    ).dt.normalize()

    statements = statements.sort_values("date")
    treasury = treasury.sort_values("date")

    hdbg.dassert(statements["date"].is_unique)
    hdbg.dassert(treasury["date"].is_unique)

    treasury["previous_trading_date"] = (
        treasury["date"].shift(1)
    )

    treasury["yield_previous_close"] = (
        treasury["yield_2yr"].shift(1)
    )

    treasury["yield_event_close"] = treasury["yield_2yr"]

    treasury["yield_change_bps"] = (
        (
            treasury["yield_event_close"]
            - treasury["yield_previous_close"]
        ) * 100
    ).round(2)

    treasury = treasury.rename(
        columns={"date": "yield_observation_date"}
    )

    yield_columns = [
        "yield_observation_date",
        "previous_trading_date",
        "yield_previous_close",
        "yield_event_close",
        "yield_change_bps",
    ]

    aligned = pd.merge_asof(
        statements,
        treasury[yield_columns],
        left_on="date",
        right_on="yield_observation_date",
        direction="forward",
        tolerance=pd.Timedelta(days=4),
    )

    aligned["same_day_yield"] = (
        aligned["date"]
        == aligned["yield_observation_date"]
    )

    hdbg.dassert_eq(len(aligned), len(statements))
    hdbg.dassert(aligned["date"].is_unique)
    hdbg.dassert(aligned["yield_change_bps"].notna().all())

    return aligned.reset_index(drop=True)


# #############################################################################
# FOMC statement text processing
# #############################################################################


def clean_statement_text(text: str) -> str:
    """
    Remove publication metadata from FOMC statement text.

    :param text: downloaded FOMC statement
    :return: cleaned monetary policy statement
    """
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove standalone publication dates.
        if re.fullmatch(
            r"[A-Za-z]+ \d{1,2}, \d{4}",
            line,
        ):
            continue

        # Remove publication and sharing metadata.
        if re.fullmatch(
            r"For immediate release(?:\s+Share)?",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        if line.lower() == "share":
            continue

        lines.append(line)

    cleaned = "\n".join(lines)

    hdbg.dassert(len(cleaned) >= 300)

    return cleaned


def split_sentences(text: str) -> list[str]:
    """
    Split an FOMC statement into individual sentences.

    :param text: monetary policy statement
    :return: nonempty sentence strings
    """
    text = clean_statement_text(text)

    # Protect common abbreviations during sentence splitting.
    abbreviations = {
        "U.S.": "U<DOT>S<DOT>",
        "Mr.": "Mr<DOT>",
        "Mrs.": "Mrs<DOT>",
        "Ms.": "Ms<DOT>",
        "Dr.": "Dr<DOT>",
    }

    for abbreviation, replacement in abbreviations.items():
        text = text.replace(abbreviation, replacement)

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [
        sentence.replace("<DOT>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]

    hdbg.dassert(len(sentences) > 0)

    return sentences


# #############################################################################
# FOMC statement tone analysis
# #############################################################################


@weave.op
def score_statement_tone(
    model: TfidfLogRegModel,
    statement_text: str,
) -> dict[str, Any]:
    """
    Calculate monetary policy tone from sentence predictions.

    :param model: fitted FOMC classification model
    :param statement_text: complete FOMC statement
    :return: sentence counts, label shares, and tone score
    """
    sentences = split_sentences(statement_text)
    labels = model.predict_batch(sentences)

    n_sentences = len(sentences)

    hdbg.dassert_eq(len(labels), n_sentences)
    hdbg.dassert(set(labels).issubset(LABEL_NAMES))

    n_dovish = labels.count(0)
    n_hawkish = labels.count(1)
    n_neutral = labels.count(2)

    dovish_share = n_dovish / n_sentences
    hawkish_share = n_hawkish / n_sentences
    neutral_share = n_neutral / n_sentences

    tone = hawkish_share - dovish_share

    hdbg.dassert(-1.0 <= tone <= 1.0)

    return {
        "n_sentences": n_sentences,
        "n_dovish": n_dovish,
        "n_hawkish": n_hawkish,
        "n_neutral": n_neutral,
        "dovish_share": dovish_share,
        "hawkish_share": hawkish_share,
        "neutral_share": neutral_share,
        "tone": tone,
    }


def calculate_statement_tones(
    aligned: pd.DataFrame,
    models: dict[str, TfidfLogRegModel],
) -> pd.DataFrame:
    """
    Calculate statement tone and tone changes for each model.

    :param aligned: FOMC statements aligned with Treasury yields
    :param models: named fitted classification models
    :return: announcement-level tone and yield change dataframe
    """
    hdbg.dassert(len(models) > 0)

    aligned = aligned.sort_values("date").reset_index(drop=True)

    records = []

    for row in aligned.itertuples(index=False):
        record = {
            "date": row.date,
            "yield_change_bps": row.yield_change_bps,
        }

        for name, model in models.items():
            result = score_statement_tone(
                model,
                row.statement_text,
            )

            record["n_sentences"] = result["n_sentences"]
            record[f"{name}_tone"] = result["tone"]

        records.append(record)

    tones = pd.DataFrame(records)

    for name in models:
        tones[f"{name}_tone_change"] = (
            tones[f"{name}_tone"].diff()
        )

    hdbg.dassert_eq(len(tones), len(aligned))
    hdbg.dassert(tones["date"].is_unique)

    return tones

# #############################################################################
# FOMC tone and Treasury yield correlation analysis
# #############################################################################


def calculate_tone_correlations(
    tones: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """
    Calculate Spearman correlations between tone and yield changes.

    :param tones: announcement-level tone and yield data
    :param model_names: classification model names
    :return: correlation results for each model
    """
    columns = [
        f"{name}_tone_change" for name in model_names
    ]

    valid = tones.dropna(
        subset=columns + ["yield_change_bps"]
    )

    hdbg.dassert(len(valid) > 2)

    rows = []

    for name in model_names:
        result = spearmanr(
            valid[f"{name}_tone_change"],
            valid["yield_change_bps"],
        )

        rows.append({
            "model": name,
            "n_observations": len(valid),
            "spearman_rho": float(result.statistic),
            "p_value": float(result.pvalue),
        })

    return pd.DataFrame(rows).set_index("model")

def bootstrap_correlation_difference(
    tones: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Estimate uncertainty in the difference between correlations.

    :param tones: announcement-level tone and yield data
    :param model_a: first classification model
    :param model_b: second classification model
    :param n_bootstrap: number of bootstrap samples
    :param random_state: random number generator seed
    :return: observed difference and confidence interval
    """
    columns = [
        f"{model_a}_tone_change",
        f"{model_b}_tone_change",
        "yield_change_bps",
    ]

    valid = tones.dropna(subset=columns)

    x_a = valid[columns[0]].to_numpy()
    x_b = valid[columns[1]].to_numpy()
    y = valid[columns[2]].to_numpy()

    n = len(valid)

    hdbg.dassert(n > 2)
    hdbg.dassert(n_bootstrap > 0)

    rho_a = float(spearmanr(x_a, y).statistic)
    rho_b = float(spearmanr(x_b, y).statistic)

    observed_difference = rho_b - rho_a

    rng = np.random.default_rng(random_state)

    differences = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        sample_a = spearmanr(
            x_a[indices],
            y[indices],
        ).statistic

        sample_b = spearmanr(
            x_b[indices],
            y[indices],
        ).statistic

        if np.isfinite(sample_a) and np.isfinite(sample_b):
            differences.append(sample_b - sample_a)

    hdbg.dassert(len(differences) > 0)

    ci_lower, ci_upper = np.percentile(
        differences,
        [2.5, 97.5],
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n_observations": n,
        "rho_a": rho_a,
        "rho_b": rho_b,
        "observed_difference": observed_difference,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_bootstrap": n_bootstrap,
        "n_valid_bootstrap": len(differences),
    }

def plot_tone_correlations(
    tones: pd.DataFrame,
    model_names: list[str],
    figsize: tuple[int, int] = (12, 5),
):
    """
    Plot tone changes against Treasury yield changes.

    :param tones: announcement-level tone and yield data
    :param model_names: classification model names
    :param figsize: figure dimensions
    :return: matplotlib figure
    """
    correlations = calculate_tone_correlations(
        tones,
        model_names,
    )

    columns = [
        f"{name}_tone_change" for name in model_names
    ]

    valid = tones.dropna(
        subset=columns + ["yield_change_bps"]
    )

    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for ax, name in zip(axes[0], model_names):
        rho = correlations.loc[name, "spearman_rho"]
        n = correlations.loc[name, "n_observations"]

        sns.regplot(
            data=valid,
            x=f"{name}_tone_change",
            y="yield_change_bps",
            ax=ax,
            ci=None,
            scatter_kws={"alpha": 0.65},
        )

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            linewidth=0.8,
        )

        ax.axvline(
            0,
            linestyle="--",
            color="gray",
            linewidth=0.8,
        )

        ax.set_title(
            f"{name.capitalize()}\n"
            f"Spearman rho = {rho:.3f}, n = {n}"
        )

        ax.set_xlabel("Change in FOMC tone")
        ax.set_ylabel("2-year yield change (bps)")

    fig.suptitle(
        "FOMC Tone Changes and Treasury Yield Changes"
    )

    fig.tight_layout()

    return fig