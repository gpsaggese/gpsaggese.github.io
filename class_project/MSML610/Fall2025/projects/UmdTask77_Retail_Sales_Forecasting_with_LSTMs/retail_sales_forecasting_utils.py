"""
Shared utilities for the Retail Sales Forecasting with LSTMs project.

Notebook code should call the functions defined here to keep cells declarative
and ensure reusability across the API and example tutorials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

_LOG = logging.getLogger(__name__)


class FeatureGenerator(Protocol):
    """Protocol for callables that receive and return pandas DataFrames."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform the provided DataFrame and return the augmented result."""


@dataclass(frozen=True)
class DataSourceConfig:
    """Container for local dataset locations and core schema metadata."""

    root_dir: Path
    sales_file: str
    calendar_file: str
    oil_file: str
    transactions_file: Optional[str]
    id_columns: Sequence[str]
    date_column: str
    target_column: str
    frequency: str
    horizon_days: int
    allow_synthetic: bool = True

    def sales_path(self) -> Path:
        """Compute the absolute path to the primary sales data file."""
        path = self.root_dir / self.sales_file
        _LOG.debug("Resolved sales path to %s", path)
        return path


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Toggle switches governing seasonal and event-based feature generation."""

    seasonalities: Sequence[int] = field(default_factory=lambda: (7, 28, 365))
    include_holidays: bool = True
    include_promotions: bool = True
    include_external_regressors: bool = False
    log_target: bool = True
    scale_numeric_features: bool = True
    lookback_days: int = 120
    train_ratio: float = 0.8


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameter bundle for the JAX recurrent neural network models."""

    cell_type: str = "lstm"  # one of {"lstm", "gru"}
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 128
    epochs: int = 20
    seed: int = 0


@dataclass
class WindowedDataset:
    """Sliding-window dataset used for training and evaluation."""

    inputs: np.ndarray  # shape: (num_samples, lookback, feature_dim)
    targets: np.ndarray  # shape: (num_samples, horizon)
    ids: pd.DataFrame    # per-sample metadata (store/product + prediction window)


@dataclass
class ForecastArtifacts:
    """Outputs from the training pipeline used for evaluation and deployment."""

    metrics: MutableMapping[str, Mapping[str, float]]
    predictions: pd.DataFrame
    model_params: Mapping[str, object]
    metadata: Dict[str, object]


def ensure_data_root(data_cfg: DataSourceConfig) -> None:
    """
    Validate dataset files and log helpful guidance when they are missing.

    When synthetic data is allowed (default) the function only warns; otherwise it
    raises `FileNotFoundError`.
    """
    required_files = {
        "sales": data_cfg.sales_path(),
        "calendar": data_cfg.root_dir / data_cfg.calendar_file,
        "oil": data_cfg.root_dir / data_cfg.oil_file,
    }
    if data_cfg.transactions_file:
        required_files["transactions"] = data_cfg.root_dir / data_cfg.transactions_file
    missing = [
        (name, path) for name, path in required_files.items() if not path.exists()
    ]
    if missing and not data_cfg.allow_synthetic:
        missing_str = ", ".join(f"{name} -> {path}" for name, path in missing)
        raise FileNotFoundError(
            f"Required dataset files are missing: {missing_str}. "
            "Set `allow_synthetic=True` to auto-generate demo data."
        )
    for name, path in missing:
        _LOG.warning("Missing %s file at %s. Synthetic fallback will be used.", name, path)
    for name, path in required_files.items():
        if path.exists():
            _LOG.debug("Found %s file at %s", name, path)


# ---------------------------------------------------------------------------
# Data ingestion and feature engineering
# ---------------------------------------------------------------------------


def _load_sales_dataframe(
    data_cfg: DataSourceConfig,
    feature_cfg: TemporalFeatureConfig,
) -> pd.DataFrame:
    path = data_cfg.sales_path()
    if path.exists():
        _LOG.info("Loading sales data from %s", path)
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame[data_cfg.date_column] = pd.to_datetime(frame[data_cfg.date_column])
        return frame
    if not data_cfg.allow_synthetic:
        raise FileNotFoundError(f"Sales file not found at {path}")
    _LOG.info("Generating synthetic sales dataset for experimentation.")
    return _generate_synthetic_sales(data_cfg, feature_cfg)


def _generate_synthetic_sales(
    data_cfg: DataSourceConfig,
    feature_cfg: TemporalFeatureConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stores = [f"store_{idx}" for idx in range(1, 5)]
    families = [f"family_{idx}" for idx in range(1, 5)]
    periods = feature_cfg.lookback_days + data_cfg.horizon_days + 120
    dates = pd.date_range(
        start=pd.Timestamp("2017-01-01"),
        periods=periods,
        freq=data_cfg.frequency,
    )
    records: List[Dict[str, object]] = []
    for store in stores:
        for family in families:
            base_level = rng.uniform(30, 80)
            weekly_amp = rng.uniform(5, 20)
            yearly_amp = rng.uniform(10, 25)
            promo_effect = rng.uniform(10, 25)
            noise_scale = rng.uniform(2, 6)
            oil_level = rng.uniform(40, 70)
            values = []
            for idx, date in enumerate(dates):
                seasonal_week = weekly_amp * np.sin(2 * np.pi * idx / 7.0)
                seasonal_year = yearly_amp * np.sin(2 * np.pi * idx / 365.0)
                promo = int((idx % 30) < 5)
                holiday = int(date.month == 12 and date.day in (24, 25, 31))
                oil_price = oil_level + np.sin(idx / 14.0) * 5 + rng.normal(0, 0.5)
                transactions = 200 + np.sin(idx / 3.0) * 25 + rng.normal(0, 5)
                sales = max(
                    0.0,
                    base_level + seasonal_week + seasonal_year + promo * promo_effect + holiday * 30
                    + rng.normal(0.0, noise_scale),
                )
                records.append(
                    {
                        data_cfg.date_column: date,
                        data_cfg.id_columns[0]: store,
                        data_cfg.id_columns[1]: family,
                        data_cfg.target_column: sales,
                        "onpromotion": promo,
                        "is_holiday": holiday,
                        "oil_price": oil_price,
                        "transactions": transactions,
                    }
                )
                values.append(sales)
    frame = pd.DataFrame(records)
    frame.sort_values(list(data_cfg.id_columns) + [data_cfg.date_column], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_feature_pipeline(
    data_cfg: DataSourceConfig,
    feature_cfg: TemporalFeatureConfig,
) -> Iterable[FeatureGenerator]:
    """
    Assemble feature generators based on the requested seasonal components.
    """
    id_columns = list(data_cfg.id_columns)
    date_column = data_cfg.date_column
    target_column = data_cfg.target_column

    def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame.sort_values(id_columns + [date_column], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        dates = output[date_column]
        output["day_of_week"] = dates.dt.dayofweek.astype(float)
        output["week_of_year"] = dates.dt.isocalendar().week.astype(int).astype(float)
        output["month"] = dates.dt.month.astype(float)
        output["day_of_month"] = dates.dt.day.astype(float)
        output["day_of_year"] = dates.dt.dayofyear.astype(float)
        for seasonality in feature_cfg.seasonalities:
            angle = 2 * np.pi * output["day_of_year"] / float(seasonality)
            output[f"seasonal_sin_{seasonality}"] = np.sin(angle)
            output[f"seasonal_cos_{seasonality}"] = np.cos(angle)
        output["is_weekend"] = (output["day_of_week"] >= 5).astype(float)
        return output

    def _join_calendar(frame: pd.DataFrame) -> pd.DataFrame:
        path = data_cfg.root_dir / data_cfg.calendar_file
        if path.exists():
            calendar = pd.read_csv(path)
            calendar[date_column] = pd.to_datetime(calendar[date_column])
            calendar["is_holiday"] = calendar.get("type", "").astype(str).isin(
                ["Holiday", "Additional", "Event", "Transfer"]
            ).astype(float)
            calendar["is_workday"] = (~calendar["is_holiday"].astype(bool)).astype(float)
            calendar = calendar[[date_column, "is_holiday", "is_workday"]]
            merged = frame.merge(calendar, on=date_column, how="left")
            merged[["is_holiday", "is_workday"]] = merged[
                ["is_holiday", "is_workday"]
            ].fillna(0.0)
            return merged
        _LOG.warning("Calendar file %s not found. Approximating holidays.", path)
        output = frame.copy()
        output["is_holiday"] = (
            (output[date_column].dt.month == 12) & (output[date_column].dt.day <= 26)
        ).astype(float)
        output["is_workday"] = 1.0 - output["is_holiday"]
        return output

    def _join_external_regressors(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        path = data_cfg.root_dir / data_cfg.oil_file
        if path.exists():
            oil = pd.read_csv(path)
            oil[date_column] = pd.to_datetime(oil[date_column])
            oil = oil.rename(columns={"dcoilwtico": "oil_price"})
            output = output.merge(oil[[date_column, "oil_price"]], on=date_column, how="left")
        else:
            _LOG.warning("Oil price file %s not found. Filling with zeros.", path)
        if "oil_price" not in output.columns:
            output["oil_price"] = 0.0
        if data_cfg.transactions_file:
            trx_path = data_cfg.root_dir / data_cfg.transactions_file
            if trx_path.exists():
                trx = pd.read_csv(trx_path)
                trx[date_column] = pd.to_datetime(trx[date_column])
                output = output.merge(
                    trx[[date_column, data_cfg.id_columns[0], "transactions"]],
                    on=[date_column, data_cfg.id_columns[0]],
                    how="left",
                )
        if "transactions" not in output.columns:
            output["transactions"] = 0.0
        return output

    def _ensure_promotions(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if "onpromotion" not in output.columns:
            output["onpromotion"] = 0.0
        output["promotion_flag"] = output["onpromotion"].astype(float)
        return output

    def _add_rollups(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for window in (7, 28):
            roll = (
                output.groupby(id_columns)[target_column]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype(float)
            )
            output[f"{target_column}_rolling_mean_{window}"] = roll
        return output

    def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = frame.select_dtypes(include=["number", "bool"]).columns
        frame = frame.copy()
        frame[numeric_cols] = frame[numeric_cols].astype(float).fillna(0.0)
        return frame

    pipeline: List[FeatureGenerator] = [_sort_frame, _add_time_features]
    if feature_cfg.include_holidays:
        pipeline.append(_join_calendar)
    if feature_cfg.include_external_regressors:
        pipeline.append(_join_external_regressors)
    if feature_cfg.include_promotions:
        pipeline.append(_ensure_promotions)
    pipeline.append(_add_rollups)
    pipeline.append(_finalize)
    _LOG.info(
        "Feature pipeline assembled with %d steps (holidays=%s, promotions=%s, external=%s).",
        len(pipeline),
        feature_cfg.include_holidays,
        feature_cfg.include_promotions,
        feature_cfg.include_external_regressors,
    )
    return pipeline


def prepare_dataloader(
    data_cfg: DataSourceConfig,
    feature_generators: Iterable[FeatureGenerator],
    feature_cfg: TemporalFeatureConfig,
) -> Tuple[WindowedDataset, WindowedDataset, Dict[str, object]]:
    """
    Prepare train/validation datasets and normalization metadata.
    """
    frame = _load_sales_dataframe(data_cfg, feature_cfg)
    for generator in feature_generators:
        frame = generator(frame)

    id_columns = list(data_cfg.id_columns)
    date_column = data_cfg.date_column
    target_column = data_cfg.target_column

    feature_columns = [
        col
        for col in frame.columns
        if col not in id_columns + [date_column, target_column, "holiday_name"]
        and frame[col].dtype.kind in ("i", "f")
    ]

    feature_values = frame[feature_columns].to_numpy(dtype=np.float32)
    feature_mean = feature_values.mean(axis=0)
    feature_std = feature_values.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    if feature_cfg.scale_numeric_features:
        frame[feature_columns] = (feature_values - feature_mean) / feature_std
    else:
        feature_mean = np.zeros_like(feature_mean)
        feature_std = np.ones_like(feature_std)

    target_array = frame[target_column].to_numpy(dtype=np.float32)
    if feature_cfg.log_target:
        frame[target_column] = np.log1p(target_array)
        target_transform = "log1p"
    else:
        target_transform = "identity"

    horizon = data_cfg.horizon_days
    lookback = feature_cfg.lookback_days
    freq = pd.tseries.frequencies.to_offset(data_cfg.frequency)

    inputs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    meta_records: List[Dict[str, object]] = []

    for keys, group in frame.groupby(id_columns):
        group = group.sort_values(date_column)
        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].to_numpy(dtype=np.float32)
        dates = group[date_column].to_numpy()
        if len(group) <= lookback + horizon:
            continue
        for start_idx in range(len(group) - lookback - horizon + 1):
            end_idx = start_idx + lookback
            horizon_end = end_idx + horizon
            inputs_list.append(features[start_idx:end_idx])
            targets_list.append(targets[end_idx:horizon_end])
            record = {col: val for col, val in zip(id_columns, keys)}
            first_forecast_day = pd.Timestamp(dates[end_idx])
            record["prediction_start"] = first_forecast_day
            record["prediction_end"] = pd.Timestamp(dates[horizon_end - 1])
            meta_records.append(record)

    if not inputs_list:
        raise ValueError("Unable to assemble training windows. Check dataset size.")

    inputs = np.stack(inputs_list, axis=0)
    targets = np.stack(targets_list, axis=0)
    ids_df = pd.DataFrame(meta_records)

    num_samples = inputs.shape[0]
    split_idx = max(1, int(num_samples * feature_cfg.train_ratio))
    train_ds = WindowedDataset(
        inputs=inputs[:split_idx],
        targets=targets[:split_idx],
        ids=ids_df.iloc[:split_idx].reset_index(drop=True),
    )
    val_ds = WindowedDataset(
        inputs=inputs[split_idx:] if split_idx < num_samples else inputs[-1:],
        targets=targets[split_idx:] if split_idx < num_samples else targets[-1:],
        ids=ids_df.iloc[split_idx:].reset_index(drop=True)
        if split_idx < num_samples
        else ids_df.tail(1).reset_index(drop=True),
    )

    metadata: Dict[str, object] = {
        "feature_columns": feature_columns,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_transform": target_transform,
        "id_columns": id_columns,
        "date_column": date_column,
        "target_column": target_column,
        "frequency": data_cfg.frequency,
        "horizon": horizon,
        "lookback": lookback,
    }
    _LOG.info(
        "Prepared datasets with %d training and %d validation windows (feature_dim=%d, horizon=%d).",
        train_ds.inputs.shape[0],
        val_ds.inputs.shape[0],
        train_ds.inputs.shape[-1],
        horizon,
    )
    return train_ds, val_ds, metadata


# ---------------------------------------------------------------------------
# JAX-based LSTM forecaster
# ---------------------------------------------------------------------------


def create_rnn_model(model_cfg: ModelConfig) -> Dict[str, object]:
    """
    Return a configuration placeholder for the handcrafted JAX LSTM.

    Parameters are instantiated during training via `_init_model_params`.
    """
    return {"model_cfg": model_cfg}


def _init_model_params(
    rng_key: jax.Array,
    feature_dim: int,
    horizon: int,
    model_cfg: ModelConfig,
) -> Dict[str, object]:
    in_dim = feature_dim
    layers: List[Dict[str, jax.Array]] = []
    key = rng_key
    for _ in range(model_cfg.num_layers):
        key, k1, k2 = jax.random.split(key, 3)
        Wx = jax.random.normal(k1, (in_dim, 4 * model_cfg.hidden_size)) * 0.03
        Wh = jax.random.normal(k2, (model_cfg.hidden_size, 4 * model_cfg.hidden_size)) * 0.03
        b = jnp.zeros((4 * model_cfg.hidden_size,))
        layers.append(
            {
                "Wx": Wx,
                "Wh": Wh,
                "b": b,
                "hidden_size": model_cfg.hidden_size,
            }
        )
        in_dim = model_cfg.hidden_size
    key, k_out = jax.random.split(key)
    W_out = jax.random.normal(k_out, (model_cfg.hidden_size, horizon)) * (
        1.0 / np.sqrt(model_cfg.hidden_size)
    )
    b_out = jnp.zeros((horizon,))
    return {"layers": tuple(layers), "output": {"W": W_out, "b": b_out}}


def _lstm_layer_step(params: Mapping[str, jax.Array], carry, x_t):
    h, c = carry
    gates = (
        jnp.matmul(x_t, params["Wx"])
        + jnp.matmul(h, params["Wh"])
        + params["b"]
    )
    i, f, g, o = jnp.split(gates, 4, axis=-1)
    i = jax.nn.sigmoid(i)
    f = jax.nn.sigmoid(f)
    g = jnp.tanh(g)
    o = jax.nn.sigmoid(o)
    new_c = f * c + i * g
    new_h = o * jnp.tanh(new_c)
    return (new_h, new_c), new_h


def _run_lstm_layers(params: Mapping[str, object], inputs: jax.Array) -> jax.Array:
    outputs = jnp.swapaxes(inputs, 0, 1)  # time-major
    for layer_params in params["layers"]:
        hidden_size = layer_params["hidden_size"]
        batch = outputs.shape[1]
        init_carry = (
            jnp.zeros((batch, hidden_size)),
            jnp.zeros((batch, hidden_size)),
        )
        (_, _), outputs = jax.lax.scan(
            lambda carry, x: _lstm_layer_step(layer_params, carry, x),
            init_carry,
            outputs,
        )
    final_hidden = outputs[-1]
    return final_hidden


def _predict(params: Mapping[str, object], inputs: jax.Array) -> jax.Array:
    hidden = _run_lstm_layers(params, inputs)
    preds = jnp.matmul(hidden, params["output"]["W"]) + params["output"]["b"]
    return preds


def train_model(
    train_ds: WindowedDataset,
    val_ds: WindowedDataset,
    model_cfg: ModelConfig,
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """
    Train the handcrafted LSTM using Optax optimizers.
    """
    rng = jax.random.PRNGKey(model_cfg.seed)
    feature_dim = train_ds.inputs.shape[-1]
    horizon = train_ds.targets.shape[-1]
    params = _init_model_params(rng, feature_dim, horizon, model_cfg)

    tx = optax.adamw(learning_rate=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay)
    opt_state = tx.init(params)

    train_inputs = jnp.asarray(train_ds.inputs)
    train_targets = jnp.asarray(train_ds.targets)
    val_inputs = jnp.asarray(val_ds.inputs)
    val_targets = jnp.asarray(val_ds.targets)

    @jax.jit
    def loss_fn(params, batch_inputs, batch_targets):
        preds = _predict(params, batch_inputs)
        loss = jnp.mean((preds - batch_targets) ** 2)
        return loss

    @jax.jit
    def train_step(params, opt_state, batch_inputs, batch_targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_inputs, batch_targets)
        if model_cfg.gradient_clip > 0:
            grads = jax.tree_map(
                lambda g: jnp.clip(g, -model_cfg.gradient_clip, model_cfg.gradient_clip),
                grads,
            )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, batch_inputs, batch_targets):
        return loss_fn(params, batch_inputs, batch_targets)

    history: List[Dict[str, float]] = []
    num_samples = train_inputs.shape[0]
    for epoch in range(model_cfg.epochs):
        rng, perm_key = jax.random.split(rng)
        permutation = np.array(jax.random.permutation(perm_key, num_samples))
        shuffled_inputs = train_inputs[permutation]
        shuffled_targets = train_targets[permutation]
        batch_losses: List[float] = []
        for start in range(0, num_samples, model_cfg.batch_size):
            end = start + model_cfg.batch_size
            batch_inputs = shuffled_inputs[start:end]
            batch_targets = shuffled_targets[start:end]
            params, opt_state, batch_loss = train_step(
                params,
                opt_state,
                batch_inputs,
                batch_targets,
            )
            batch_losses.append(float(batch_loss))
        train_loss = float(np.mean(batch_losses))
        val_loss = float(eval_step(params, val_inputs, val_targets))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        _LOG.debug(
            "Epoch %d/%d | train_loss=%.5f | val_loss=%.5f",
            epoch + 1,
            model_cfg.epochs,
            train_loss,
            val_loss,
        )

    return {
        "params": params,
        "opt_state": opt_state,
        "history": history,
        "model_cfg": model_cfg,
        "metadata": metadata,
    }


def _inverse_target_transform(values: np.ndarray, metadata: Mapping[str, object]) -> np.ndarray:
    if metadata.get("target_transform") == "log1p":
        return np.expm1(values)
    return values


def _compute_metric(predictions: np.ndarray, targets: np.ndarray, metric: str) -> float:
    metric = metric.lower()
    if metric == "mae":
        return float(np.mean(np.abs(predictions - targets)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(predictions - targets))))
    if metric == "mape":
        denom = np.maximum(np.abs(targets), 1e-3)
        return float(np.mean(np.abs((targets - predictions) / denom)) * 100)
    raise ValueError(f"Unsupported metric: {metric}")


def _aggregate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    ids: pd.DataFrame,
    metrics: Sequence[str],
    id_columns: Sequence[str],
) -> MutableMapping[str, Mapping[str, float]]:
    results: MutableMapping[str, Dict[str, float]] = {}
    for metric in metrics:
        metric_lower = metric.lower()
        values: Dict[str, float] = {}
        values["overall"] = _compute_metric(predictions, targets, metric_lower)
        grouped = ids.groupby(list(id_columns))
        for key, index in grouped.groups.items():
            key_preds = predictions[list(index)]
            key_targets = targets[list(index)]
            key_value = _compute_metric(key_preds, key_targets, metric_lower)
            key_str = "/".join(map(str, key)) if isinstance(key, tuple) else str(key)
            values[key_str] = key_value
        results[metric_lower] = values
    return results


def _build_predictions_dataframe(
    predictions: np.ndarray,
    targets: np.ndarray,
    ids: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    horizon = predictions.shape[1]
    freq = pd.tseries.frequencies.to_offset(metadata["frequency"])
    rows: List[Dict[str, object]] = []
    for idx, row in ids.iterrows():
        start_date: pd.Timestamp = row["prediction_start"]
        for step in range(horizon):
            timestamp = start_date + step * freq
            record = {col: row[col] for col in metadata["id_columns"]}
            record["timestamp"] = timestamp
            record["prediction"] = float(predictions[idx, step])
            record["target"] = float(targets[idx, step])
            rows.append(record)
    return pd.DataFrame(rows)


def evaluate_model(
    trained_state: Mapping[str, object],
    validation_ds: WindowedDataset,
    metadata: Mapping[str, object],
    metrics: Sequence[str],
) -> ForecastArtifacts:
    """
    Compute metrics and generate tidy prediction outputs for the validation set.
    """
    params = trained_state["params"]
    raw_predictions = np.array(_predict(params, jnp.asarray(validation_ds.inputs)))
    raw_targets = np.array(validation_ds.targets)

    predictions = _inverse_target_transform(raw_predictions, metadata)
    targets = _inverse_target_transform(raw_targets, metadata)

    metrics_dict = _aggregate_metrics(
        predictions,
        targets,
        validation_ds.ids,
        metrics,
        metadata["id_columns"],
    )
    predictions_df = _build_predictions_dataframe(predictions, targets, validation_ds.ids, metadata)

    return ForecastArtifacts(
        metrics=metrics_dict,
        predictions=predictions_df,
        model_params=params,
        metadata={
            "history": trained_state.get("history", []),
            "feature_columns": metadata["feature_columns"],
            "horizon": metadata["horizon"],
        },
    )

