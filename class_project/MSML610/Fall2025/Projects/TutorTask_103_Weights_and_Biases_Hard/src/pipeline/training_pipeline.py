from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import traceback

from src.utils.config import config_manager
from src.logging.logger import WandbLogger
from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.preprocessor import Preprocessor
from src.components.model_registry import ModelRegistry


class TrainingPipeline:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.models_to_run = self.params["training"].get("models_to_run", [])
        self.logger = wandb_logger or WandbLogger(config_path)
        self.ingestor = DataIngestion(config_path, self.logger)
        self.fe = FeatureEngineering(config_path, self.logger)
        self.prep = Preprocessor(config_path, self.logger)
        self.trainer = ModelTrainer(config_path, self.logger)
        self.evaluator = ModelEvaluation(config_path, self.logger)  
        self.registry = ModelRegistry(config_path, self.logger)

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
        trained_models: Dict[str, Any] = {}
        for model_name in self.models_to_run:
            # Many models are optional in the professor Docker image (prophet, lightgbm, xgboost, tensorflow).
            # For Phase 5/6, we prefer completing the full pipeline and recording "SKIPPED" rather than hard-failing.
            try:
                if model_name == "lstm":
                    try:
                        tr = self.trainer.train_lstm_keras_tuner(
                            data["X_train_seq"], data["y_train_seq"], data["X_val_seq"], data["y_val_seq"]
                        )
                        test_metrics = self.evaluator.evaluate(
                            tr.model, data["X_test_seq"], data["y_test_seq"], tr.model_name
                        )
                        results["lstm"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                        trained_models[tr.model_name] = tr.model
                    except ImportError as e:
                        self.logger.warning(f"Skipping LSTM: {e}")
                        results["lstm"] = "SKIPPED: missing tensorflow/keras-tuner"
                    except Exception as e:
                        # Capture traceback for Phase 6 reporting.
                        tb = traceback.format_exc()
                        self.logger.error(f"LSTM failed: {e}\n{tb}")
                        artifacts_dir = Path(self.params.get("evaluation", {}).get("artifacts_dir", "artifacts"))
                        (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
                        (artifacts_dir / "metrics" / "lstm_error.txt").write_text(tb)
                        results["lstm"] = f"ERROR: {e}"
                elif model_name == "linear_regression":
                    tr = self.trainer.train_linear_regression(
                        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
                    )
                    test_metrics = self.evaluator.evaluate(tr.model, data["X_test"], data["y_test"], tr.model_name)
                    results["linear_regression"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "xgboost":
                    tr = self.trainer.train_xgboost(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
                    test_metrics = self.evaluator.evaluate(tr.model, data["X_test"], data["y_test"], tr.model_name)
                    results["xgboost"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "lightgbm":
                    tr = self.trainer.train_lightgbm(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
                    test_metrics = self.evaluator.evaluate(tr.model, data["X_test"], data["y_test"], tr.model_name)
                    results["lightgbm"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "random_forest":
                    tr = self.trainer.train_random_forest(
                        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
                    )
                    test_metrics = self.evaluator.evaluate(tr.model, data["X_test"], data["y_test"], tr.model_name)
                    results["random_forest"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "ma":
                    tr = self.trainer.train_ma(data["y_train_s"], data["y_val_s"])
                    test_metrics = self.evaluator.evaluate(tr.model, None, data["y_test_s"], tr.model_name)
                    results["ma"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "ar":
                    tr = self.trainer.train_ar(data["y_train_s"], data["y_val_s"])
                    test_metrics = self.evaluator.evaluate(tr.model, None, data["y_test_s"], tr.model_name)
                    results["ar"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "arima":
                    tr = self.trainer.train_arima(data["y_train_s"], data["y_val_s"])
                    test_metrics = self.evaluator.evaluate(tr.model, None, data["y_test_s"], tr.model_name)
                    results["arima"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "sarimax":
                    tr = self.trainer.train_sarimax(
                        data["y_train_s"], data["y_val_s"], data["X_train_df"], data["X_val_df"]
                    )
                    test_metrics = self.evaluator.evaluate(tr.model, data["X_test_df"], data["y_test_s"], tr.model_name)
                    results["sarimax"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                    trained_models[tr.model_name] = tr.model
                elif model_name == "prophet":
                    tr = self.trainer.train_prophet(data["y_train_s"], data["y_val_s"])
                    X_test_ds = pd.DataFrame({"ds": pd.to_datetime(data["y_test_s"].index)})
                    test_metrics = self.evaluator.evaluate(tr.model, X_test_ds, data["y_test_s"], tr.model_name)
                    results["prophet"] = {"val_metrics": tr.metrics, "test_metrics": test_metrics}
                elif model_name == "cnn":
                    results["cnn"] = "TODO: call CNN trainer with seq data"
                else:
                    self.logger.warning(f"Unknown model '{model_name}', skipping.")
                    continue
            except ImportError as e:
                self.logger.warning(f"Skipping {model_name}: {e}")
                results[model_name] = f"SKIPPED: {e}"
                continue

        # 5) Ensemble (stacking) - train meta-model on validation preds only (no test leakage).
        ens_cfg = (self.params.get("training", {}) or {}).get("ensemble", {}) or {}
        if bool(ens_cfg.get("enabled", False)) and str(ens_cfg.get("type", "stacking")) == "stacking":
            base_models = list(ens_cfg.get("base_models", []))
            available = [m for m in base_models if m in trained_models]
            if len(available) >= 2:
                try:
                    from sklearn.linear_model import LinearRegression
                    import numpy as np
                except Exception:
                    available = []

            if len(available) >= 2:
                X_val_stack = np.column_stack([trained_models[m].predict(data["X_val"]) for m in available])
                X_test_stack = np.column_stack([trained_models[m].predict(data["X_test"]) for m in available])

                meta = LinearRegression()
                meta.fit(X_val_stack, data["y_val"])

                class _StackingModel:
                    def __init__(self, base, meta_model):
                        self._base = base
                        self._meta = meta_model

                    def predict(self, X):
                        Xs = np.column_stack([self._base[m].predict(X) for m in available])
                        return self._meta.predict(Xs)

                ensemble_model = _StackingModel(trained_models, meta)
                ens_test_metrics = self.evaluator.evaluate(ensemble_model, data["X_test"], data["y_test"], "ensemble_stacking")
                results["ensemble_stacking"] = {
                    "base_models": available,
                    "test_metrics": ens_test_metrics,
                }
                self.logger.info(f"Ensemble stacking using base_models={available}: {ens_test_metrics}")
            else:
                self.logger.warning("Ensemble enabled but not enough base models were trained; skipping ensemble.")

        # Persist metrics to disk so you can compare models without relying on W&B UI.
        metrics_dir = Path(self.params["evaluation"].get("artifacts_dir", "artifacts")) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        primary = str(self.params.get("training", {}).get("metrics_primary", "RMSE"))
        best = {"model": None, "metric": primary, "value": None}
        for name, res in results.items():
            if not isinstance(res, dict):
                continue
            val = (res.get("val_metrics") or {}).get(primary)
            if val is None:
                continue
            if best["value"] is None or float(val) < float(best["value"]):
                best = {"model": name, "metric": primary, "value": float(val)}

        out = {
            "timestamp_utc": ts,
            "ticker": self.params.get("data_collection", {}).get("ticker_symbol"),
            "models_to_run": list(self.models_to_run),
            "results": results,
            "best_by_val": best,
        }
        (metrics_dir / "last_run.json").write_text(json.dumps(out, indent=2))
        (metrics_dir / f"run_{ts}.json").write_text(json.dumps(out, indent=2))

        # Phase 4: finalize and save the best model (retrain on train+val, then evaluate on test).
        if best["model"] == "linear_regression":
            try:
                import numpy as np
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            except Exception:
                self.logger.warning("Could not import sklearn/numpy to finalize best model; skipping save_best_only.")
            else:
                X_tv = np.vstack([data["X_train"], data["X_val"]])
                y_tv = np.concatenate([data["y_train"], data["y_val"]])
                final_model = LinearRegression()
                final_model.fit(X_tv, y_tv)
                final_test_metrics = self.evaluator.evaluate(final_model, data["X_test"], data["y_test"], "linear_regression_final")
                self.registry.save(
                    "linear_regression_final",
                    final_model,
                    scaler=data.get("scaler"),
                    metadata={
                        "trained_on": "train+val",
                        "primary_metric": primary,
                        "best_by_val": best,
                        "final_test_metrics": final_test_metrics,
                        "feature_names": data.get("feature_names"),
                    },
                )
                results["linear_regression_final"] = {"test_metrics": final_test_metrics}

        # Finish W&B
        self.logger.finish()
        return {"data": data, "results": results}


def run_pipeline():
    pipe = TrainingPipeline()
    return pipe.run()


if __name__ == "__main__":
    run_pipeline()