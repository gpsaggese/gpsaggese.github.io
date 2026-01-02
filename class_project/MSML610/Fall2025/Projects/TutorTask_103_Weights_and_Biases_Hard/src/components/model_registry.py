import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


@dataclass
class SavedModelInfo:
    model_name: str
    model_path: Path
    scaler_path: Optional[Path]
    metadata_path: Path


class ModelRegistry:
    """
    Save the chosen/best model and related artifacts to disk (and optionally to W&B).

    We keep this small and pragmatic: save model via joblib, save metadata JSON,
    and save scaler if provided.
    """

    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.logger = wandb_logger or WandbLogger(config_path)
        self.artifacts_dir = Path(self.params["evaluation"].get("artifacts_dir", "artifacts"))
        self.out_dir = self.artifacts_dir / "best_model"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_name: str,
        model: Any,
        scaler: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SavedModelInfo:
        model_path = self.out_dir / f"{model_name}.joblib"
        metadata_path = self.out_dir / f"{model_name}_meta.json"
        scaler_path = self.out_dir / f"{model_name}_scaler.joblib" if scaler is not None else None

        joblib.dump(model, model_path)
        if scaler_path is not None:
            joblib.dump(scaler, scaler_path)

        meta = metadata or {}
        meta.update({"model_name": model_name})
        metadata_path.write_text(json.dumps(meta, indent=2))

        self.logger.info(f"Saved best model to {model_path}")
        if self.logger.run:
            self.logger.log_artifact(str(model_path), f"best_model_{model_name}", "model")
            self.logger.log_artifact(str(metadata_path), f"best_model_{model_name}_meta", "metadata")
            if scaler_path is not None:
                self.logger.log_artifact(str(scaler_path), f"best_model_{model_name}_scaler", "preprocessor")

        return SavedModelInfo(model_name=model_name, model_path=model_path, scaler_path=scaler_path, metadata_path=metadata_path)

