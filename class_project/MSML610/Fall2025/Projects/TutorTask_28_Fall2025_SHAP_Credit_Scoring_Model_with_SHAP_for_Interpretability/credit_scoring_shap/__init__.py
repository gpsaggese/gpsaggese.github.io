"""
Top-level convenience imports for the credit_scoring_shap package.

This makes the notebooks nicer to read:

    from credit_scoring_shap import (
        DataConfig, ModelConfig, TrainingConfig,
        load_raw_data, load_and_preprocess,
        build_model, train_model, evaluate_model,
        plot_roc_curves, plot_confusion_matrix,
        build_shap_explainer,
        plot_global_shap_summary,
        plot_shap_dependence_for_top_feature,
        plot_shap_decision_for_index,
        run_sensitivity_for_instance,
    )
"""

from .config import DataConfig, ModelConfig, TrainingConfig
from .data import load_raw_data, load_and_preprocess
from .modeling import build_model, train_model, evaluate_model
from .evaluation import plot_roc_curves, plot_confusion_matrix, save_metrics_text
from .explain import (
    build_shap_explainer,
    plot_global_shap_summary,
    plot_shap_dependence_for_top_feature,
    plot_shap_decision_for_index,
)
from .sensitivity import run_sensitivity_for_instance

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_raw_data",
    "load_and_preprocess",
    "build_model",
    "train_model",
    "evaluate_model",
    "plot_roc_curves",
    "plot_confusion_matrix",
    "save_metrics_text",
    "build_shap_explainer",
    "plot_global_shap_summary",
    "plot_shap_dependence_for_top_feature",
    "plot_shap_decision_for_index",
    "run_sensitivity_for_instance",
]
