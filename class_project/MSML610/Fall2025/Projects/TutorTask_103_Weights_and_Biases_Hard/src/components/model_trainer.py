# src/components/model_trainer.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

# --- add this new method inside ModelTrainer class (below train_linear_regression) ---
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        cfg = self.params.get("model", {}).get("random_forest", {})
        training_cfg = self.params.get("training", {})
        random_state = int(training_cfg.get("random_state", 42))

        n_estimators = int(cfg.get("n_estimators", 400))
        max_depth = cfg.get("max_depth", None)
        max_features = cfg.get("max_features", "sqrt")

        # sklearn may warn about "auto"; map it safely
        if max_features == "auto":
            max_features = 1.0

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-9))) * 100.0)
        r2 = float(r2_score(y_val, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        self.logger.info(f"RandomForest validation metrics: {metrics}")

        if self.logger.run:
            self.logger.log_metrics({f"random_forest/{k}": v for k, v in metrics.items()})

        return TrainResult(model_name="random_forest", metrics=metrics, model=model)

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "xgboost is required. Inside the Docker container (venv active) run:\n"
                "  pip install xgboost\n"
            ) from e

        cfg = self.params.get("model", {}).get("xgboost", {})
        training_cfg = self.params.get("training", {})
        random_state = int(training_cfg.get("random_state", 42))

        model = XGBRegressor(
            n_estimators=int(cfg.get("n_estimators", 400)),
            max_depth=int(cfg.get("max_depth", 5)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        # XGBoost's sklearn API has changed across versions (and the professor's Docker
        # image may ship a different one). Some versions reject eval_metric/evals_result
        # in `.fit()`. We therefore pass only kwargs supported by the installed version.
        evals_result: Dict[str, Dict[str, list]] = {}
        try:
            import inspect

            fit_sig = inspect.signature(model.fit)
            fit_kwargs: Dict[str, Any] = {}
            if "eval_set" in fit_sig.parameters:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
            if "eval_metric" in fit_sig.parameters:
                fit_kwargs["eval_metric"] = "rmse"
            if "verbose" in fit_sig.parameters:
                fit_kwargs["verbose"] = False
            if "evals_result" in fit_sig.parameters:
                fit_kwargs["evals_result"] = evals_result

            model.fit(X_train, y_train, **fit_kwargs)
        except Exception:
            # Fall back to the simplest fit if signature inspection fails.
            model.fit(X_train, y_train)

        # Best-effort: fetch eval history if available.
        try:
            evals_result = model.evals_result()  # type: ignore[attr-defined]
        except Exception:
            pass

        # Save training curve (RMSE over boosting rounds) for reporting.
        try:
            rmse_curve = evals_result.get("validation_0", {}).get("rmse", [])
            if rmse_curve:
                artifacts_dir = Path(self.params.get("evaluation", {}).get("artifacts_dir", "artifacts"))
                plots_dir = Path(self.params.get("evaluation", {}).get("plots_dir", str(artifacts_dir / "plots")))
                plots_dir.mkdir(parents=True, exist_ok=True)
                curve_path = artifacts_dir / "metrics" / "xgboost_rmse_curve.csv"
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"iteration": list(range(1, len(rmse_curve) + 1)), "val_rmse": rmse_curve}).to_csv(
                    curve_path, index=False
                )
                import matplotlib.pyplot as plt

                plt.figure(figsize=(7, 4))
                plt.plot(range(1, len(rmse_curve) + 1), rmse_curve, label="val_rmse")
                plt.xlabel("Boosting round")
                plt.ylabel("RMSE")
                plt.title("XGBoost training curve (validation RMSE)")
                plt.tight_layout()
                plt.savefig(plots_dir / "xgboost_val_rmse_curve.png", dpi=200)
                plt.close()
                if self.logger.run:
                    # log final rmse + the curve file as an artifact-like table
                    self.logger.log_metrics({"xgboost/val_rmse_last": float(rmse_curve[-1])})
        except Exception:
            # Never fail training due to plotting.
            pass

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-9))) * 100.0)
        r2 = float(r2_score(y_val, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        self.logger.info(f"XGBoost validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"xgboost/{k}": v for k, v in metrics.items()})

        return TrainResult(model_name="xgboost", metrics=metrics, model=model)

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "lightgbm is required. Inside the Docker container (venv active) run:\n"
                "  pip install lightgbm\n"
            ) from e

        cfg = self.params.get("model", {}).get("lightgbm", {})
        training_cfg = self.params.get("training", {})
        random_state = int(training_cfg.get("random_state", 42))

        model = lgb.LGBMRegressor(
            n_estimators=int(cfg.get("n_estimators", 400)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            num_leaves=int(cfg.get("num_leaves", 63)),
            max_depth=int(cfg.get("max_depth", -1)),
            feature_fraction=float(cfg.get("feature_fraction", 0.9)),
            bagging_fraction=float(cfg.get("bagging_fraction", 0.9)),
            bagging_freq=int(cfg.get("bagging_freq", 1)),
            # Avoid std::random_device usage issues inside locked-down containers.
            deterministic=True,
            seed=random_state,
            feature_fraction_seed=random_state,
            bagging_seed=random_state,
            data_random_seed=random_state,
            random_state=random_state,
            n_jobs=-1,
        )
        try:
            import lightgbm as lgb  # type: ignore
        except Exception:
            lgb = None  # type: ignore

        evals_result: Dict[str, Dict[str, list]] = {}
        callbacks = []
        if lgb is not None:
            callbacks = [lgb.record_evaluation(evals_result)]  # type: ignore

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # Save training curve (RMSE over boosting rounds) for reporting.
        try:
            # record_evaluation stores under evals_result["valid_0"]["rmse"]
            rmse_curve = evals_result.get("valid_0", {}).get("rmse", [])
            if rmse_curve:
                artifacts_dir = Path(self.params.get("evaluation", {}).get("artifacts_dir", "artifacts"))
                plots_dir = Path(self.params.get("evaluation", {}).get("plots_dir", str(artifacts_dir / "plots")))
                plots_dir.mkdir(parents=True, exist_ok=True)
                curve_path = artifacts_dir / "metrics" / "lightgbm_rmse_curve.csv"
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"iteration": list(range(1, len(rmse_curve) + 1)), "val_rmse": rmse_curve}).to_csv(
                    curve_path, index=False
                )
                import matplotlib.pyplot as plt

                plt.figure(figsize=(7, 4))
                plt.plot(range(1, len(rmse_curve) + 1), rmse_curve, label="val_rmse")
                plt.xlabel("Boosting round")
                plt.ylabel("RMSE")
                plt.title("LightGBM training curve (validation RMSE)")
                plt.tight_layout()
                plt.savefig(plots_dir / "lightgbm_val_rmse_curve.png", dpi=200)
                plt.close()
                if self.logger.run:
                    self.logger.log_metrics({"lightgbm/val_rmse_last": float(rmse_curve[-1])})
        except Exception:
            pass

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + 1e-9))) * 100.0)
        r2 = float(r2_score(y_val, y_pred))

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        self.logger.info(f"LightGBM validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"lightgbm/{k}": v for k, v in metrics.items()})

        return TrainResult(model_name="lightgbm", metrics=metrics, model=model)

    # ---------------------------------------------------------------------
    # Statistical time-series models (statsmodels)
    # ---------------------------------------------------------------------
    def train_ma(self, y_train: "pd.Series", y_val: "pd.Series") -> TrainResult:
        """
        Moving Average model = ARIMA(0, d, q). We'll use d=0 by default.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("statsmodels is required. Install inside Docker: pip install statsmodels") from e

        cfg = self.params.get("model", {}).get("ma", {})
        q = int(cfg.get("q", 5))
        d = int(cfg.get("d", 0))

        model = ARIMA(y_train.astype(float), order=(0, d, q))
        fitted = model.fit()
        wrapper = _StatsForecastWrapper(fitted, uses_exog=False)

        y_pred = wrapper.predict(steps=len(y_val))
        y_val_arr = np.asarray(y_val, dtype=float)
        metrics = {
            "MAE": float(mean_absolute_error(y_val_arr, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val_arr, y_pred))),
            "MAPE": _mape(y_val_arr, y_pred),
            "R2": float(r2_score(y_val_arr, y_pred)),
        }
        self.logger.info(f"MA validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"ma/{k}": v for k, v in metrics.items()})
        return TrainResult(model_name="ma", metrics=metrics, model=wrapper)

    def train_ar(self, y_train: "pd.Series", y_val: "pd.Series") -> TrainResult:
        """
        AutoRegressive model = ARIMA(p, d, 0). We'll use d=0 by default.
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("statsmodels is required. Install inside Docker: pip install statsmodels") from e

        cfg = self.params.get("model", {}).get("ar", {})
        p = int(cfg.get("p", 5))
        d = int(cfg.get("d", 0))

        model = ARIMA(y_train.astype(float), order=(p, d, 0))
        fitted = model.fit()
        wrapper = _StatsForecastWrapper(fitted, uses_exog=False)

        y_pred = wrapper.predict(steps=len(y_val))
        y_val_arr = np.asarray(y_val, dtype=float)
        metrics = {
            "MAE": float(mean_absolute_error(y_val_arr, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val_arr, y_pred))),
            "MAPE": _mape(y_val_arr, y_pred),
            "R2": float(r2_score(y_val_arr, y_pred)),
        }
        self.logger.info(f"AR validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"ar/{k}": v for k, v in metrics.items()})
        return TrainResult(model_name="ar", metrics=metrics, model=wrapper)

    def train_arima(self, y_train: "pd.Series", y_val: "pd.Series") -> TrainResult:
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("statsmodels is required. Install inside Docker: pip install statsmodels") from e

        cfg = self.params.get("model", {}).get("arima", {})
        order = tuple(cfg.get("order", [5, 1, 0]))
        order = (int(order[0]), int(order[1]), int(order[2]))

        model = ARIMA(y_train.astype(float), order=order)
        fitted = model.fit()
        wrapper = _StatsForecastWrapper(fitted, uses_exog=False)

        y_pred = wrapper.predict(steps=len(y_val))
        y_val_arr = np.asarray(y_val, dtype=float)
        metrics = {
            "MAE": float(mean_absolute_error(y_val_arr, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val_arr, y_pred))),
            "MAPE": _mape(y_val_arr, y_pred),
            "R2": float(r2_score(y_val_arr, y_pred)),
        }
        self.logger.info(f"ARIMA validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"arima/{k}": v for k, v in metrics.items()})
        return TrainResult(model_name="arima", metrics=metrics, model=wrapper)

    def train_sarimax(
        self,
        y_train: "pd.Series",
        y_val: "pd.Series",
        X_train: "pd.DataFrame",
        X_val: "pd.DataFrame",
    ) -> TrainResult:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("statsmodels is required. Install inside Docker: pip install statsmodels") from e

        cfg = self.params.get("model", {}).get("arima", {})
        order = tuple(cfg.get("order", [5, 1, 0]))
        order = (int(order[0]), int(order[1]), int(order[2]))
        seasonal = tuple(cfg.get("seasonal_order", [0, 0, 0, 0]))
        seasonal = (int(seasonal[0]), int(seasonal[1]), int(seasonal[2]), int(seasonal[3]))

        model = SARIMAX(
            endog=y_train.astype(float),
            exog=X_train.astype(float),
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        wrapper = _StatsForecastWrapper(fitted, uses_exog=True)

        y_pred = wrapper.predict(X_val.astype(float), steps=len(y_val))
        y_val_arr = np.asarray(y_val, dtype=float)
        metrics = {
            "MAE": float(mean_absolute_error(y_val_arr, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val_arr, y_pred))),
            "MAPE": _mape(y_val_arr, y_pred),
            "R2": float(r2_score(y_val_arr, y_pred)),
        }
        self.logger.info(f"SARIMAX validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"sarimax/{k}": v for k, v in metrics.items()})
        return TrainResult(model_name="sarimax", metrics=metrics, model=wrapper)

    # ---------------------------------------------------------------------
    # Prophet (optional)
    # ---------------------------------------------------------------------
    def train_prophet(self, y_train: "pd.Series", y_val: "pd.Series") -> TrainResult:
        try:
            from prophet import Prophet  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("prophet is required. Install inside Docker: pip install prophet") from e

        cfg = self.params.get("model", {}).get("prophet", {})
        model = Prophet(
            yearly_seasonality=bool(cfg.get("yearly_seasonality", True)),
            weekly_seasonality=bool(cfg.get("weekly_seasonality", True)),
            daily_seasonality=bool(cfg.get("daily_seasonality", False)),
            seasonality_mode=str(cfg.get("seasonality_mode", "multiplicative")),
            changepoint_prior_scale=float(cfg.get("changepoint_prior_scale", 0.05)),
        )

        df_train = pd.DataFrame({"ds": pd.to_datetime(y_train.index), "y": y_train.astype(float).values})
        model.fit(df_train)

        df_val = pd.DataFrame({"ds": pd.to_datetime(y_val.index)})
        y_pred = model.predict(df_val)["yhat"].values

        y_val_arr = np.asarray(y_val, dtype=float)
        metrics = {
            "MAE": float(mean_absolute_error(y_val_arr, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_val_arr, y_pred))),
            "MAPE": _mape(y_val_arr, y_pred),
            "R2": float(r2_score(y_val_arr, y_pred)),
        }
        self.logger.info(f"Prophet validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"prophet/{k}": v for k, v in metrics.items()})

        class _ProphetWrapper:
            def __init__(self, m):
                self._m = m

            def predict(self, X):
                # X must be a DataFrame with column 'ds'
                return self._m.predict(X)["yhat"].values

        return TrainResult(model_name="prophet", metrics=metrics, model=_ProphetWrapper(model))

    # ---------------------------------------------------------------------
    # Deep learning: LSTM with optional KerasTuner BayesianOptimization
    # ---------------------------------------------------------------------
    def train_lstm_keras_tuner(
        self,
        X_train_seq: np.ndarray,
        y_train_seq: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_seq: np.ndarray,
    ) -> TrainResult:
        """
        Train LSTM with KerasTuner BayesianOptimization (if available).
        Falls back to a single configured model if keras_tuner/tensorflow are missing.
        """
        try:
            import tensorflow as tf  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "TensorFlow is required for LSTM. In the professor container this may not be available on Python 3.12.\n"
                "If TF import fails, we will switch to a PyTorch LSTM implementation."
            ) from e

        try:
            import keras_tuner as kt  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "keras-tuner is required for LSTM tuning. Install inside Docker:\n"
                "  pip install keras-tuner"
            ) from e

        cfg = self.params.get("model", {}).get("lstm", {})
        tuning_cfg = self.params.get("hyperparameter_tuning", {})
        max_trials = int(tuning_cfg.get("max_runs", 20))
        # Allow runtime overrides to avoid long / memory-heavy runs inside the professor container.
        max_trials = int(os.environ.get("LSTM_MAX_TRIALS", str(max_trials)))
        seed = int(self.params.get("training", {}).get("random_state", 42))
        tf.keras.utils.set_random_seed(seed)
        tf.keras.backend.clear_session()

        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

        def build_model(hp: "kt.HyperParameters") -> "tf.keras.Model":
            tf.keras.backend.clear_session()
            units1 = hp.Choice("lstm_units_1", [50, 64, 100], default=int(cfg.get("lstm_units", [50, 50])[0]))
            units2 = hp.Choice("lstm_units_2", [0, 50, 64], default=int(cfg.get("lstm_units", [50, 50])[1]))
            dropout = hp.Float("dropout_rate", 0.0, 0.5, step=0.1, default=float(cfg.get("dropout_rate", 0.2)))
            dense_units = hp.Choice("dense_units", [16, 25, 32, 64], default=int(cfg.get("dense_units", 25)))
            lr = hp.Choice("learning_rate", [0.0005, 0.001, 0.002], default=float(cfg.get("learning_rate", 0.001)))

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=input_shape))
            model.add(tf.keras.layers.LSTM(units1, return_sequences=(units2 != 0)))
            model.add(tf.keras.layers.Dropout(dropout))
            if units2 != 0:
                model.add(tf.keras.layers.LSTM(units2))
                model.add(tf.keras.layers.Dropout(dropout))
            model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
            model.add(tf.keras.layers.Dense(1))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="mse",
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
            )
            return model

        tuner = kt.BayesianOptimization(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=max_trials,
            directory="artifacts/tuning",
            project_name="lstm_kt",
            overwrite=True,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ]

        batch_size = int(os.environ.get("LSTM_BATCH_SIZE", str(int(cfg.get("batch_size", 32)))))
        epochs = int(os.environ.get("LSTM_EPOCHS", str(int(cfg.get("epochs", 30)))))
        tuner.search(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Refit best model once to capture a clean epoch-by-epoch history for reporting.
        best_hp = tuner.get_best_hyperparameters(1)[0]
        best_model = build_model(best_hp)

        history = best_model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Save history (CSV + plot) for Phase 6.
        try:
            artifacts_dir = Path(self.params.get("evaluation", {}).get("artifacts_dir", "artifacts"))
            plots_dir = Path(self.params.get("evaluation", {}).get("plots_dir", str(artifacts_dir / "plots")))
            plots_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
            hist_df = pd.DataFrame(history.history)
            hist_df.insert(0, "epoch", list(range(1, len(hist_df) + 1)))
            hist_df.to_csv(artifacts_dir / "metrics" / "lstm_history.csv", index=False)

            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            if "loss" in hist_df.columns:
                plt.plot(hist_df["epoch"], hist_df["loss"], label="train_loss")
            if "val_loss" in hist_df.columns:
                plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.title("LSTM training curve (loss)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "lstm_loss_curve.png", dpi=200)
            plt.close()
        except Exception:
            pass

        # Log per-epoch metrics to W&B.
        if self.logger.run:
            try:
                for i in range(len(history.history.get("loss", []))):
                    payload = {}
                    for k, v in history.history.items():
                        try:
                            payload[f"lstm/{k}"] = float(v[i])
                        except Exception:
                            continue
                    if payload:
                        self.logger.log_metrics(payload, step=i + 1)
            except Exception:
                pass

        y_pred = best_model.predict(X_val_seq, verbose=0).reshape(-1)

        mae = float(mean_absolute_error(y_val_seq, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val_seq, y_pred)))
        mape = float(np.mean(np.abs((y_val_seq - y_pred) / (y_val_seq + 1e-9))) * 100.0)
        r2 = float(r2_score(y_val_seq, y_pred))
        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

        self.logger.info(f"LSTM (KerasTuner) validation metrics: {metrics}")
        if self.logger.run:
            self.logger.log_metrics({f"lstm/{k}": v for k, v in metrics.items()})

        class _KerasWrapper:
            def __init__(self, m):
                self._m = m
            def predict(self, X):
                return self._m.predict(X, verbose=0).reshape(-1)

        return TrainResult(model_name="lstm", metrics=metrics, model=_KerasWrapper(best_model))


class _StatsForecastWrapper:
    """
    A thin wrapper around statsmodels fitted results to match the .predict(X) interface.
    - For non-exogenous models (AR/MA/ARIMA): ignore X, use steps based on len(X) or explicit steps.
    - For SARIMAX: X is required as exog for forecasting.
    """

    def __init__(self, fitted: Any, uses_exog: bool):
        self._fitted = fitted
        self._uses_exog = uses_exog

    def predict(self, X: Any = None, steps: Optional[int] = None) -> np.ndarray:
        if steps is None:
            if X is None:
                raise ValueError("Provide steps or X with a length.")
            steps = len(X)
        exog = None
        if self._uses_exog:
            if X is None:
                raise ValueError("SARIMAX prediction requires exogenous features X.")
            exog = X
        return np.asarray(self._fitted.forecast(steps=int(steps), exog=exog))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)

