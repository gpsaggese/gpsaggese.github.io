"""Semantic Kernel plugin for housing price analysis."""

import json

from semantic_kernel.functions import kernel_function

from housing_data import load_housing_data
from housing_models import train_models


class HousingAnalysisPlugin:
    """Expose housing analysis functions to Semantic Kernel."""

    @kernel_function(
        name="dataset_summary",
        description="Get information about the housing dataset."
    )
    def dataset_summary(self) -> str:
        X, y = load_housing_data()

        result = {
            "rows": len(X),
            "features": list(X.columns),
            "target": y.name,
            "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
        }
        return json.dumps(result)

    @kernel_function(
        name="compare_models",
        description="Train Linear Regression and Random Forest and compare scores."
    )
    def compare_models(self) -> str:
        _, comparison = train_models()

        result = {
            model_name: {
                metric: float(value)
                for metric, value in scores.items()
            }
            for model_name, scores in comparison.to_dict(
                orient="index"
            ).items()
        }

        return json.dumps(result)
