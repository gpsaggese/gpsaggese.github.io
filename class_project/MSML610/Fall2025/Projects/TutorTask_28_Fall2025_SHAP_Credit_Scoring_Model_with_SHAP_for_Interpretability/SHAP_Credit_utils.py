"""
High-level utilities for the SHAP credit scoring project.

This module glues together the credit_scoring_shap package into
simple functions that are easy to call from notebooks.
"""

from typing import Dict, Any, Optional

from credit_scoring_shap.config import TrainingConfig
from credit_scoring_shap import (
    load_and_preprocess,
    build_model,
    train_model,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    build_shap_explainer,
    plot_global_shap_summary,
    plot_shap_dependence_for_top_feature,
    plot_shap_decision_for_index,
    run_sensitivity_for_instance,
    save_metrics_text,
)


def run_full_pipeline(cfg: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """
    Run an end-to-end experiment:
      * load + preprocess data
      * train XGBoost
      * evaluate metrics
      * compute SHAP values
      * generate plots + sensitivity for one good and one bad example

    Returns:
        A dictionary with key objects and metrics.
    """
    cfg = cfg or TrainingConfig()
    cfg.ensure_reports_dir()

    # 1) Data
    X_train, X_test, y_train, y_test, preproc, feature_names = load_and_preprocess(
        cfg.data
    )

    # 2) Model training
    model = build_model(cfg.model)
    model = train_model(model, X_train, y_train)

    # 3) Evaluation
    metrics, y_proba, y_pred = evaluate_model(model, X_test, y_test)
    save_metrics_text(metrics, cfg)
    plot_confusion_matrix(metrics["confusion_matrix"], cfg)
    plot_roc_curves(y_test, y_proba, cfg)

    # 4) SHAP
    explainer, shap_values = build_shap_explainer(model, X_train)
    plot_global_shap_summary(shap_values, X_train, cfg)
    top_feature = plot_shap_dependence_for_top_feature(shap_values, X_train, cfg)

    # pick one good + one bad example if possible
    good_idx = None
    bad_idx = None
    for i, y in enumerate(y_test.values):
        if y == 1 and good_idx is None:
            good_idx = i
        if y == 0 and bad_idx is None:
            bad_idx = i
        if good_idx is not None and bad_idx is not None:
            break

    if good_idx is not None:
        plot_shap_decision_for_index(
            explainer, shap_values, X_test, good_idx, cfg, prefix="good"
        )
        run_sensitivity_for_instance(
            model, X_test, shap_values, cfg, good_idx, top_n=3
        )

    if bad_idx is not None:
        plot_shap_decision_for_index(
            explainer, shap_values, X_test, bad_idx, cfg, prefix="bad"
        )
        run_sensitivity_for_instance(
            model, X_test, shap_values, cfg, bad_idx, top_n=3
        )

    return {
        "config": cfg,
        "model": model,
        "preprocessor": preproc,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "explainer": explainer,
        "shap_values": shap_values,
        "top_feature": top_feature,
        "good_idx": good_idx,
        "bad_idx": bad_idx,
    }
