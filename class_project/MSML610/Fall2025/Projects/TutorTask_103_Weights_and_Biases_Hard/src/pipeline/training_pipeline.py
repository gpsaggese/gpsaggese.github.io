# src/pipeline/training_pipeline.py
from typing import Optional, Dict, Any

from src.utils.config import config_manager
from src.logging.logger import WandbLogger
from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.preprocessor import Preprocessor


class TrainingPipeline:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.models_to_run = self.params["training"].get("models_to_run", [])
        self.logger = wandb_logger or WandbLogger(config_path)
        self.ingestor = DataIngestion(config_path, self.logger)
        self.fe = FeatureEngineering(config_path, self.logger)
        self.prep = Preprocessor(config_path, self.logger)

    def run(self) -> Dict[str, Any]:
        # Start W&B run if not already started
        if not self.logger.run:
            self.logger.init_run(run_name=self.params["wandb"].get("experiment_name"))

        # 1) Ingestion
        df_raw = self.ingestor.run(name="stock")

        # 2) Feature engineering
        df_feats = self.fe.run(df_raw, name="stock")

        # 3) Preprocessing (flat + sequences)
        data = self.prep.run(df_feats)

        # 4) Model training dispatch (stubs)
        results = {}
        for model_name in self.models_to_run:
            if model_name == "lstm":
                results["lstm"] = "TODO: call LSTM trainer with seq data"
            elif model_name == "linear_regression":
                results["linear_regression"] = "TODO: call LR trainer with flat data"
            elif model_name == "xgboost":
                results["xgboost"] = "TODO: call XGBoost trainer with flat data"
            elif model_name == "lightgbm":
                results["lightgbm"] = "TODO: call LightGBM trainer with flat data"
            elif model_name == "random_forest":
                results["random_forest"] = "TODO: call RF trainer with flat data"
            elif model_name == "cnn":
                results["cnn"] = "TODO: call CNN trainer with seq data"
            else:
                self.logger.warning(f"Unknown model '{model_name}', skipping.")
                continue

        # Finish W&B
        self.logger.finish()
        return {"data": data, "results": results}


def run_pipeline():
    pipe = TrainingPipeline()
    return pipe.run()


if __name__ == "__main__":
    run_pipeline()