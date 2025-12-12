# src/components/model_evaluation.py
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


class ModelEvaluation:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.eval_cfg = self.params["evaluation"]
        self.logger = wandb_logger or WandbLogger(config_path)

        self.artifacts_dir = Path(self.eval_cfg["artifacts_dir"])
        self.plots_dir = Path(self.eval_cfg["plots_dir"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)
        r2 = float(r2_score(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def plot_actual_vs_pred(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Path:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual", linewidth=1)
        plt.plot(y_pred, label="Predicted", linewidth=1)
        plt.title(f"Actual vs Predicted ({name})")
        plt.legend()
        plt.tight_layout()
        out = self.plots_dir / f"{name}_actual_vs_pred.png"
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Path:
        residuals = y_true - y_pred
        fig = plt.figure(figsize=(10, 4))
        plt.plot(residuals, linewidth=1)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"Residuals ({name})")
        plt.tight_layout()
        out = self.plots_dir / f"{name}_residuals.png"
        plt.savefig(out, dpi=200)
        plt.close(fig)
        return out

    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, name: str) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        metrics = self.compute_metrics(y_test, y_pred)

        self.logger.info(f"{name} test metrics: {metrics}")

        avp = self.plot_actual_vs_pred(y_test, y_pred, name)
        res = self.plot_residuals(y_test, y_pred, name)

        if self.logger.run:
            self.logger.log_metrics({f"{name}/test_{k}": v for k, v in metrics.items()})
            # If your WandbLogger only supports matplotlib figs, we can log images later.
            # For now, plots are saved to artifacts/plots.

        return metrics