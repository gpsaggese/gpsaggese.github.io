"""
Lightweight regression tests for the retail sales forecasting utilities.

These tests rely on the synthetic dataset fallback so they can run in CI without
requiring Kaggle credentials or large downloads.
"""

import sys
from pathlib import Path

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retail_sales_forecasting_utils import (
    TrainingConfig,
    load_run_metrics,
    prepare_dataset,
    run_training_pipeline,
    train_models,
)


def _demo_config(data_root: Path) -> TrainingConfig:
    """Create a lightweight config that forces synthetic fallback."""
    return TrainingConfig(
        data_dir=data_root,
        families=("GROCERY I",),
        max_stores=2,
        context_length=10,
        horizon=4,
        epochs=1,
        batch_size=32,
        synthetic_if_missing=True,
    )


def test_prepare_dataset_shapes(tmp_path: Path) -> None:
    """Preparing datasets with synthetic fallback should yield non-empty windows."""
    cfg = _demo_config(tmp_path / "data")
    dataset = prepare_dataset(cfg)
    assert dataset.train_inputs.ndim == 3
    assert dataset.train_targets.ndim == 2
    assert dataset.train_inputs.shape[0] > 0
    assert dataset.val_inputs.shape[0] > 0
    assert dataset.train_targets.shape[-1] == cfg.horizon
    assert dataset.sales_scaler.std != 0


def test_run_training_pipeline(tmp_path: Path) -> None:
    """Smoke test the end-to-end training and artifact persistence."""
    cfg = _demo_config(tmp_path / "data")
    run_dir, results, dataset = run_training_pipeline(
        cfg,
        output_dir=tmp_path / "runs",
        run_name="test_run",
        model_names=("lstm",),
        rng=jax.random.PRNGKey(0),
    )
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "lstm_metrics.json").exists()
    assert len(results) == 1
    assert results[0].history  # epochs logged
    metrics = load_run_metrics(run_dir)
    assert "lstm" in metrics
    assert np.isfinite(metrics["lstm"]["normalized_metrics"]["val_mse"])
    results2, dataset2 = train_models(cfg, model_names=("lstm",), rng=jax.random.PRNGKey(1))
    assert len(results2) == 1
    assert dataset2.train_inputs.shape == dataset.train_inputs.shape
