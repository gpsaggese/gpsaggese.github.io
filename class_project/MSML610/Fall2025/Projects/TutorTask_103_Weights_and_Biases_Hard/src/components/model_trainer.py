# src/components/model_trainer.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


@dataclass
class TrainResult:
    model_name: str
    metrics: Dict[str, float]
    model: Any


class ModelTrainer:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.logger = wandb_logger or WandbLogger(config_path)

    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> TrainResult:
        cfg = self.params["model"].get("linear_regression", {})
        fit_intercept = bool(cfg.get("fit_intercept", True))

        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-9))) * 100.0)
        r2 = float(r2_score(y_val, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        self.logger.info(f"LinearRegression validation metrics: {metrics}")

        if self.logger.run:
            self.logger.log_metrics({f"linear_regression/{k}": v for k, v in metrics.items()})

        return TrainResult(model_name="linear_regression", metrics=metrics, model=model)