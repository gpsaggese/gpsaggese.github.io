"""
Executable walkthrough for the Retail Sales Forecasting project.

The script mirrors the steps documented in
`retail_sales_forecasting_with_lstms.example.md` and demonstrates how to use the
utilities module to train both LSTM/GRU models, persist artifacts, run
inference, and emit the same plots bundled in the final package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from retail_sales_forecasting_utils import (
    TrainingConfig,
    load_run_metrics,
    plot_breakdowns,
    plot_final_metrics_comparison,
    plot_predictions_sample,
    plot_sales_metrics,
    plot_training_curves,
    run_inference,
    run_training_pipeline,
)

_LOG = logging.getLogger(__name__)


def run_demo(
    data_root: Path,
    artifacts_root: Path,
    families: Sequence[str] | None = None,
) -> Path:
    """
    Train LSTM and GRU baselines, save metrics, and generate diagnostic plots.

    Parameters
    ----------
    data_root:
        Path containing the Kaggle CSV files. If the files are missing, the
        utilities fall back to a reproducible synthetic dataset.
    artifacts_root:
        Directory where the run directory plus plots will be saved.
    families:
        Optional subset of product families to keep for the quick demo.

    Returns
    -------
    Path to the saved training run directory.
    """
    default_families = TrainingConfig().families
    cfg = TrainingConfig(
        data_dir=data_root,
        families=tuple(families) if families else default_families,
        max_stores=5,
        context_length=30,
        horizon=7,
        epochs=4,
        batch_size=256,
        synthetic_if_missing=True,
    )

    _LOG.info("Launching demo training using data_dir=%s", data_root)
    run_dir, results, dataset = run_training_pipeline(
        cfg,
        output_dir=artifacts_root,
        run_name="demo_run",
        rng=None,
        model_names=("lstm", "gru"),
    )
    metrics = load_run_metrics(run_dir)
    plot_training_curves(metrics, run_dir)
    plot_final_metrics_comparison(metrics, run_dir)
    plot_sales_metrics(metrics, run_dir)
    for model in ("lstm", "gru"):
        plot_breakdowns(metrics, run_dir, model)
    inference = run_inference(run_dir, "lstm", dataset=dataset, cfg=cfg)
    plot_predictions_sample(
        inference["predictions"],
        inference["targets"],
        dataset,
        run_dir / "lstm_test_predictions.png",
        sample_index=0,
    )
    _LOG.info("Demo artifacts saved to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[0]
    data_root = (project_root / "data" / "store-sales-time-series-forecasting").resolve()
    artifacts_root = (project_root / "artifacts").resolve()
    run_demo(data_root, artifacts_root)
