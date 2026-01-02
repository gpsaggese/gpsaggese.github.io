"""
Bayesian hyperparameter tuning for RandomForest using Optuna.

Why a separate script?
- `training_pipeline.py` stays as "train once per model"
- this script runs many trials and logs metrics + writes a CSV/JSON summary to disk

How it works:
1) Load `config/params.yaml`
2) Read `hyperparameter_tuning.spaces.random_forest`
3) Optuna suggests values (categorical over the space values)
4) Train RF on train split, evaluate on val split (objective = minimize val RMSE)
5) Save all trials to `artifacts/tuning/random_forest_trials.csv` and best config to JSON

Run (offline):
  export WANDB_MODE=offline
  python scripts/tune_random_forest_bayes.py

Run (online dashboard):
  export WANDB_MODE=online
  python scripts/tune_random_forest_bayes.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import json
import time

import numpy as np

try:
    import optuna  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Optuna is required for Bayesian tuning.\n"
        "Install inside the professor Docker container (venv active):\n"
        "  pip install optuna\n"
    ) from e

from src.logging.logger import WandbLogger
from src.utils.config import config_manager
from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.preprocessor import Preprocessor
from src.components.model_trainer import ModelTrainer, TrainResult
from src.components.model_evaluation import ModelEvaluation


def _suggest_from_space(trial: "optuna.Trial", space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Treat each key in YAML space as categorical among the provided values.
    This is "Bayesian over categorical choices" (TPE sampler).
    """
    hp: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, list):
            hp[k] = trial.suggest_categorical(k, v)
        else:
            hp[k] = v
    return hp


def _train_rf_with_params(
    base_params: Dict[str, Any],
    hp: Dict[str, Any],
    data: Dict[str, Any],
    trainer: ModelTrainer,
) -> TrainResult:
    # Patch trainer.params for this trial without touching YAML on disk.
    trainer.params = deepcopy(base_params)
    trainer.params.setdefault("model", {})
    trainer.params["model"].setdefault("random_forest", {})
    trainer.params["model"]["random_forest"].update(hp)
    return trainer.train_random_forest(data["X_train"], data["y_train"], data["X_val"], data["y_val"])


def main() -> None:
    params = config_manager.load_params()
    tuning_cfg = params.get("hyperparameter_tuning", {})
    max_runs = int(tuning_cfg.get("max_runs", 20))
    space = (tuning_cfg.get("spaces", {}) or {}).get("random_forest")
    if not isinstance(space, dict):
        raise ValueError("Missing hyperparameter_tuning.spaces.random_forest in config/params.yaml")

    # Output locations (ignored by git via .gitignore)
    tuning_dir = Path(params.get("evaluation", {}).get("artifacts_dir", "artifacts")) / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = tuning_dir / "random_forest_trials.csv"
    best_json = tuning_dir / "random_forest_best.json"

    logger = WandbLogger("config")
    if not logger.run:
        logger.init_run(run_name="tune_random_forest_bayes")

    # Build dataset ONCE (this is expensive; don't redo per trial).
    ing = DataIngestion("config", logger)
    fe = FeatureEngineering("config", logger)
    prep = Preprocessor("config", logger)
    trainer = ModelTrainer("config", logger)
    evaluator = ModelEvaluation("config", logger)

    df_raw = ing.run(name="stock")
    df_feats = fe.run(df_raw, name="stock")
    data = prep.run(df_feats)

    primary = str(params.get("training", {}).get("metrics_primary", "RMSE"))
    direction = "minimize" if primary.upper() != "R2" else "maximize"

    sampler = optuna.samplers.TPESampler(seed=int(params.get("training", {}).get("random_state", 42)))
    study = optuna.create_study(direction=direction, sampler=sampler, study_name="rf_bayes")

    rows: List[Dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        hp = _suggest_from_space(trial, space)
        t0 = time.time()
        tr = _train_rf_with_params(params, hp, data, trainer)
        elapsed = time.time() - t0

        # Objective uses validation metric (no test leakage).
        score = float(tr.metrics.get(primary, np.inf))
        trial.set_user_attr("val_metrics", tr.metrics)
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("elapsed_s", elapsed)

        # Log to W&B (offline ok)
        if logger.run:
            logger.log_metrics({f"rf_tune/val_{k}": v for k, v in tr.metrics.items()})
            logger.log_metrics({"rf_tune/trial": trial.number, "rf_tune/elapsed_s": elapsed, f"rf_tune/val_{primary}": score})

        rows.append(
            {
                "trial": trial.number,
                "elapsed_s": elapsed,
                **{f"hp_{k}": v for k, v in hp.items()},
                **{f"val_{k}": v for k, v in tr.metrics.items()},
            }
        )

        return score

    study.optimize(objective, n_trials=max_runs)

    # Save trials CSV (simple, no pandas dependency needed here)
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join(str(r.get(k, "")) for k in keys))
        trials_csv.write_text("\n".join(lines) + "\n")

    best_hp = dict(study.best_trial.user_attrs.get("hp", {}))
    best_val_metrics = dict(study.best_trial.user_attrs.get("val_metrics", {}))

    # Final: evaluate best model on TEST once (for reporting).
    best_tr = _train_rf_with_params(params, best_hp, data, trainer)
    test_metrics = evaluator.evaluate(best_tr.model, data["X_test"], data["y_test"], "random_forest_best")

    out = {
        "primary_metric": primary,
        "direction": direction,
        "best_params": best_hp,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": test_metrics,
        "n_trials": max_runs,
    }
    best_json.write_text(json.dumps(out, indent=2))

    logger.info(f"Saved RF tuning trials to {trials_csv}")
    logger.info(f"Saved RF best config/metrics to {best_json}")
    if logger.run:
        logger.log_artifact(str(trials_csv), "rf_tuning_trials", "dataset")
        logger.log_artifact(str(best_json), "rf_tuning_best", "metadata")

    logger.finish()


if __name__ == "__main__":
    main()


