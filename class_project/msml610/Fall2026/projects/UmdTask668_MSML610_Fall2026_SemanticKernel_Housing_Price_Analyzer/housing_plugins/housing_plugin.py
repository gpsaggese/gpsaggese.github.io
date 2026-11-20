"""Semantic Kernel plugin for housing price analysis."""

import json
import threading
from collections.abc import Callable

import pandas as pd
from semantic_kernel.functions import kernel_function

from housing_data import load_housing_data
from housing_models import train_models

DataLoader = Callable[[], tuple[pd.DataFrame, pd.Series]]


class HousingAnalysisPlugin:
    """
    Expose housing analysis functions to Semantic Kernel.

    Results are cached per plugin instance: the dataset is loaded at most
    once, and the models are trained at most once, however many times the
    agent calls the functions. Failed loads are not cached. Use a new plugin
    instance to start from fresh data.
    """

    def __init__(self, data_loader: DataLoader = load_housing_data) -> None:
        """
        Initialize the plugin.

        :param data_loader: function returning the features `X` and the
            target `y`; defaults to downloading the UCI dataset, and can be
            replaced with a local data source (e.g., synthetic data in tests)
        """
        self._data_loader = data_loader
        self._lock = threading.Lock()
        self._data: tuple[pd.DataFrame, pd.Series] | None = None
        self._comparison: str | None = None

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the dataset, loading it on first use."""
        with self._lock:
            if self._data is None:
                self._data = self._data_loader()
            return self._data

    @kernel_function(
        name="dataset_summary",
        description=(
            "Get the housing dataset's number of rows, feature names, "
            "target name, and missing-value count."
        ),
    )
    def dataset_summary(self) -> str:
        X, y = self._load_data()

        result = {
            "rows": len(X),
            "features": list(X.columns),
            "target": y.name,
            "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
        }
        return json.dumps(result)

    @kernel_function(
        name="compare_models",
        description=(
            "Train Linear Regression and Random Forest on the same "
            "train/test split and return their test MAE, RMSE, and R2 scores."
        ),
    )
    def compare_models(self) -> str:
        if self._comparison is not None:
            return self._comparison
        X, y = self._load_data()
        _, comparison = train_models(X, y)

        result = {
            model_name: {
                metric: float(value)
                for metric, value in scores.items()
            }
            for model_name, scores in comparison.to_dict(
                orient="index"
            ).items()
        }

        self._comparison = json.dumps(result)
        return self._comparison
