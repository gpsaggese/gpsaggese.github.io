
"""Reusable utilities for financial news sentiment analysis."""

import logging
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """Initialize the project utility logger."""
    global _LOG
    _LOG = notebook_log.getChild("financial_sentiment_utils")


def normalize_headline(text: str) -> str:
    """Normalize whitespace and lowercase a financial headline."""
    if not isinstance(text, str):
        raise TypeError("Headline must be a string.")
    return " ".join(text.lower().split())


def clean_headline(text: str) -> str:
    """Remove extra whitespace while preserving financial symbols."""
    if not isinstance(text, str):
        raise TypeError("Headline must be a string.")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculate accuracy and macro F1 for sentiment predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted labels must have equal lengths.")
    if len(y_true) == 0:
        raise ValueError("Prediction arrays cannot be empty.")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def build_model_comparison(model_results: dict) -> pd.DataFrame:
    """Build a model comparison table sorted by macro F1."""
    rows = []

    for model_name, metrics in model_results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
        })

    return (
        pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro F1"])
        .sort_values("Macro F1", ascending=False)
        .reset_index(drop=True)
    )
