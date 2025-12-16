"""
Utilities for the Retail Sales Forecasting with LSTMs project (JAX).

This module consolidates the configuration dataclasses, data loaders, JAX/Flax
models, evaluation helpers, inference routines, and plotting utilities that
power both the API and example tutorials. The functions defined here are
consumed by notebooks, scripts, and tests to keep the project organized around a
single reusable surface.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import linen as nn

_LOG = logging.getLogger(__name__)


@dataclass
class ScalingStats:
    """Stores normalization parameters for log-transformed sales."""

    mean: float
    std: float

    def normalize(self, values: np.ndarray) -> np.ndarray:
        std = self.std if self.std != 0 else 1.0
        return (values - self.mean) / std

    def denormalize_sales(self, normalized: np.ndarray) -> np.ndarray:
        """Inverse transform to original sales space."""
        values = normalized * (self.std if self.std != 0 else 1.0) + self.mean
        return np.expm1(values)


@dataclass
class DatasetSplits:
    """Sliding-window datasets plus metadata for analysis."""

    train_inputs: jnp.ndarray
    train_targets: jnp.ndarray
    val_inputs: jnp.ndarray
    val_targets: jnp.ndarray
    sales_scaler: ScalingStats
    train_store_ids: Optional[np.ndarray] = None
    train_family_ids: Optional[np.ndarray] = None
    train_holiday_flags: Optional[np.ndarray] = None
    train_promo_flags: Optional[np.ndarray] = None
    val_store_ids: Optional[np.ndarray] = None
    val_family_ids: Optional[np.ndarray] = None
    val_holiday_flags: Optional[np.ndarray] = None
    val_promo_flags: Optional[np.ndarray] = None


@dataclass
class TrainingConfig:
    """Configuration controlling dataset filters and JAX model hyperparameters."""

    data_dir: Path = Path("data/store-sales-time-series-forecasting")
    families: Sequence[str] = (
        "GROCERY I",
        "BEVERAGES",
        "PRODUCE",
        "CLEANING",
        "DAIRY",
    )
    max_stores: int = 10
    start_date: str = "2015-01-01"
    split_date: str = "2017-04-01"
    context_length: int = 30
    horizon: int = 7
    hidden_size: int = 64
    learning_rate: float = 3e-3
    batch_size: int = 512
    epochs: int = 6
    seed: int = 0
    synthetic_if_missing: bool = True


DEFAULT_CFG = TrainingConfig()


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std()
    if pd.isna(std) or std == 0:
        std = 1.0
    return (values - values.mean()) / std


def _generate_synthetic_frame(cfg: TrainingConfig) -> pd.DataFrame:
    """Create a lightweight synthetic dataset when Kaggle CSVs are unavailable."""
    rng = np.random.default_rng(cfg.seed or 42)
    stores = list(range(1, min(cfg.max_stores, 5) + 1))
    families = list(cfg.families)[:5]
    num_days = cfg.context_length + cfg.horizon + 90
    dates = pd.date_range(start=pd.Timestamp(cfg.start_date), periods=num_days, freq="D")

    records: List[Dict[str, object]] = []
    for store in stores:
        for family in families:
            base = rng.uniform(40, 120)
            weekly_amp = rng.uniform(10, 25)
            annual_amp = rng.uniform(5, 15)
            promo_gain = rng.uniform(20, 60)
            noise = rng.uniform(1, 8)
            for idx, date in enumerate(dates):
                seasonal_week = weekly_amp * np.sin(2 * np.pi * idx / 7)
                seasonal_year = annual_amp * np.sin(2 * np.pi * idx / 365)
                promo = 1 if (idx % 28) < 4 else 0
                holiday = int(date.month == 12 and date.day in (24, 25, 31))
                transactions = 200 + 15 * np.sin(idx / 5.0) + rng.normal(0, 5)
                sales = max(
                    0.0,
                    base + seasonal_week + seasonal_year + promo_gain * promo + holiday * 30 + rng.normal(0, noise),
                )
                records.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": sales,
                        "onpromotion": promo,
                        "transactions": transactions,
                        "is_holiday": holiday,
                    }
                )
    frame = pd.DataFrame(records)
    frame.sort_values(["store_nbr", "family", "date"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_feature_frame(cfg: TrainingConfig) -> Tuple[pd.DataFrame, ScalingStats]:
    """Load Kaggle CSVs (or a synthetic fallback) and engineer covariates."""
    data_dir = cfg.data_dir
    train_path = data_dir / "train.csv"
    holidays_path = data_dir / "holidays_events.csv"
    transactions_path = data_dir / "transactions.csv"

    if train_path.exists():
        _LOG.info("Loading training data from %s", train_path)
        df = pd.read_csv(train_path, parse_dates=["date"])
    else:
        if not cfg.synthetic_if_missing:
            raise FileNotFoundError(f"Dataset not found at {train_path}.")
        _LOG.warning(
            "Training file %s missing. Falling back to synthetic data generation for demos.",
            train_path,
        )
        df = _generate_synthetic_frame(cfg)

    df = df[df["date"] >= pd.Timestamp(cfg.start_date)]

    if cfg.families:
        df = df[df["family"].isin(cfg.families)]

    if cfg.max_stores:
        top_stores = (
            df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(cfg.max_stores).index
        )
        df = df[df["store_nbr"].isin(top_stores)]

    # Merge transactions to capture demand surges.
    if transactions_path.exists():
        transactions = pd.read_csv(transactions_path, parse_dates=["date"])
        df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    if "transactions" not in df.columns:
        df["transactions"] = (
            df.groupby("store_nbr")["sales"].transform(lambda s: 150 + np.sqrt(np.maximum(s, 0)))
        )
    df["transactions"] = df.groupby("store_nbr")["transactions"].transform(lambda s: s.fillna(s.mean()))
    df["transactions"] = df["transactions"].fillna(df["transactions"].mean())

    # Holiday flags focus on national and regional events.
    if holidays_path.exists():
        holidays = pd.read_csv(holidays_path, parse_dates=["date"])
        holidays = holidays[holidays["type"].isin(["Holiday", "Transfer", "Additional"])]
        holidays = holidays[holidays["locale"].isin(["National", "Regional"])]
        holidays = holidays[["date"]].drop_duplicates()
        holidays["is_holiday"] = 1
        df = df.merge(holidays, on="date", how="left")
    if "is_holiday" not in df.columns:
        df["is_holiday"] = (
            df["date"].dt.month.eq(12) & df["date"].dt.day.isin([24, 25, 31])
        ).astype(int)
    df["is_holiday"] = df["is_holiday"].fillna(0)

    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(float)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["sales_log"] = np.log1p(df["sales"])
    scaler = ScalingStats(mean=float(df["sales_log"].mean()), std=float(df["sales_log"].std()))
    df["sales_norm"] = scaler.normalize(df["sales_log"].to_numpy())

    if "onpromotion" not in df.columns:
        df["onpromotion"] = 0
    df["promo_norm"] = _zscore(np.log1p(df["onpromotion"]))
    df["transactions_norm"] = _zscore(np.log1p(df["transactions"]))

    store_lookup = {store: idx for idx, store in enumerate(sorted(df["store_nbr"].unique()))}
    fam_lookup = {fam: idx for idx, fam in enumerate(sorted(df["family"].unique()))}
    store_den = max(len(store_lookup) - 1, 1)
    fam_den = max(len(fam_lookup) - 1, 1)

    df["store_scaled"] = df["store_nbr"].map(lambda s: store_lookup[s] / store_den)
    df["family_scaled"] = df["family"].map(lambda f: fam_lookup[f] / fam_den)

    feature_cols = [
        "promo_norm",
        "transactions_norm",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_holiday",
        "is_weekend",
        "store_scaled",
        "family_scaled",
    ]
    df = df.sort_values(["store_nbr", "family", "date"])
    df = df[["date", "store_nbr", "family", "sales_norm"] + feature_cols]
    return df, scaler


def create_sequences(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    sales_scaler: ScalingStats,
) -> DatasetSplits:
    """Converts time series per (store, family) into sliding windows."""
    feature_cols = [col for col in df.columns if col not in {"date", "store_nbr", "family", "sales_norm"}]
    split_ts = pd.Timestamp(cfg.split_date)
    context, horizon = cfg.context_length, cfg.horizon
    total_window = context + horizon

    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []
    train_metadata = {"store": [], "family": [], "holiday": [], "promo": []}
    val_metadata = {"store": [], "family": [], "holiday": [], "promo": []}

    for (store_nbr, family), grp in df.groupby(["store_nbr", "family"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < total_window:
            continue
        feature_matrix = grp[feature_cols].to_numpy(dtype=np.float32)
        target_series = grp["sales_norm"].to_numpy(dtype=np.float32)
        dates = grp["date"]
        # Get holiday and promo flags for target period
        holiday_flags = grp["is_holiday"].to_numpy()
        promo_flags = grp["promo_norm"].to_numpy()  # Using normalized promo as flag

        for start in range(0, len(grp) - total_window + 1):
            context_slice = feature_matrix[start : start + context]
            target_slice = target_series[start + context : start + total_window]
            target_holiday = holiday_flags[start + context : start + total_window]
            target_promo = promo_flags[start + context : start + total_window]
            
            if target_slice.shape[0] != horizon:
                raise ValueError(
                    f"Target slice length {target_slice.shape[0]} does not match horizon {horizon}."
                )
            context_end = dates.iloc[start + context - 1]
            if context_end < split_ts:
                train_inputs.append(context_slice)
                train_targets.append(target_slice)
                train_metadata["store"].append(store_nbr)
                train_metadata["family"].append(family)
                train_metadata["holiday"].append(target_holiday)
                train_metadata["promo"].append(target_promo)
            else:
                val_inputs.append(context_slice)
                val_targets.append(target_slice)
                val_metadata["store"].append(store_nbr)
                val_metadata["family"].append(family)
                val_metadata["holiday"].append(target_holiday)
                val_metadata["promo"].append(target_promo)

    def _to_array(items: Sequence[np.ndarray]) -> jnp.ndarray:
        if not items:
            return jnp.empty((0, context, len(feature_cols)), dtype=jnp.float32)
        return jnp.array(items, dtype=jnp.float32)

    dataset = DatasetSplits(
        train_inputs=_to_array(train_inputs),
        train_targets=jnp.array(train_targets, dtype=jnp.float32) if train_targets else jnp.empty(
            (0, cfg.horizon), dtype=jnp.float32
        ),
        val_inputs=_to_array(val_inputs),
        val_targets=jnp.array(val_targets, dtype=jnp.float32) if val_targets else jnp.empty(
            (0, cfg.horizon), dtype=jnp.float32
        ),
        sales_scaler=sales_scaler,
        train_store_ids=np.array(train_metadata["store"]) if train_metadata["store"] else None,
        train_family_ids=np.array(train_metadata["family"]) if train_metadata["family"] else None,
        train_holiday_flags=np.array(train_metadata["holiday"], dtype=bool) if train_metadata["holiday"] else None,
        train_promo_flags=np.array(train_metadata["promo"]) if train_metadata["promo"] else None,
        val_store_ids=np.array(val_metadata["store"]) if val_metadata["store"] else None,
        val_family_ids=np.array(val_metadata["family"]) if val_metadata["family"] else None,
        val_holiday_flags=np.array(val_metadata["holiday"], dtype=bool) if val_metadata["holiday"] else None,
        val_promo_flags=np.array(val_metadata["promo"]) if val_metadata["promo"] else None,
    )
    return dataset


class LSTMForecaster(nn.Module):
    hidden_size: int
    horizon: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Create Dense layer and initialize it before scan
        dense = nn.Dense(4 * self.hidden_size, name="lstm_kernel")
        # Initialize by calling with dummy input to register parameters
        batch_size = x.shape[0]
        dummy_input = jnp.concatenate([
            jnp.zeros((batch_size, x.shape[-1])),
            jnp.zeros((batch_size, self.hidden_size))
        ], axis=-1)
        _ = dense(dummy_input)  # Initialize layer parameters

        def step(carry, x_t):
            h, c = carry
            # Call dense layer - parameters are already registered
            gates = dense(jnp.concatenate([x_t, h], axis=-1))
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            i = nn.sigmoid(i)
            f = nn.sigmoid(f)
            o = nn.sigmoid(o)
            g = jnp.tanh(g)
            c = f * c + i * g
            h = o * jnp.tanh(c)
            return (h, c), h

        init = (
            jnp.zeros((batch_size, self.hidden_size)),
            jnp.zeros((batch_size, self.hidden_size)),
        )
        (_, _), hidden_states = jax.lax.scan(step, init, jnp.swapaxes(x, 0, 1))
        final_hidden = hidden_states[-1]
        return nn.Dense(self.horizon, name="output_proj")(final_hidden)


class GRUForecaster(nn.Module):
    hidden_size: int
    horizon: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_dense = nn.Dense(2 * self.hidden_size, name="gate_kernel")
        candidate_dense = nn.Dense(self.hidden_size, name="candidate_kernel")
        
        # Initialize layers by calling with dummy inputs
        batch_size = x.shape[0]
        dummy_gate_input = jnp.concatenate([
            jnp.zeros((batch_size, x.shape[-1])),
            jnp.zeros((batch_size, self.hidden_size))
        ], axis=-1)
        dummy_candidate_input = jnp.concatenate([
            jnp.zeros((batch_size, x.shape[-1])),
            jnp.zeros((batch_size, self.hidden_size))
        ], axis=-1)
        _ = gate_dense(dummy_gate_input)  # Initialize gate layer
        _ = candidate_dense(dummy_candidate_input)  # Initialize candidate layer

        def step(hidden, x_t):
            gates = gate_dense(jnp.concatenate([x_t, hidden], axis=-1))
            r, z = jnp.split(gates, 2, axis=-1)
            r = nn.sigmoid(r)
            z = nn.sigmoid(z)
            candidate_input = jnp.concatenate([x_t, r * hidden], axis=-1)
            h_tilde = jnp.tanh(candidate_dense(candidate_input))
            hidden = (1 - z) * h_tilde + z * hidden
            return hidden, hidden

        init_hidden = jnp.zeros((batch_size, self.hidden_size))
        _, hidden_states = jax.lax.scan(step, init_hidden, jnp.swapaxes(x, 0, 1))
        final_hidden = hidden_states[-1]
        return nn.Dense(self.horizon, name="output_proj")(final_hidden)


def build_model(name: str, cfg: TrainingConfig) -> nn.Module:
    """Factory returning the requested recurrent module."""
    name_lower = name.lower()
    if name_lower == "lstm":
        return LSTMForecaster(hidden_size=cfg.hidden_size, horizon=cfg.horizon)
    if name_lower == "gru":
        return GRUForecaster(hidden_size=cfg.hidden_size, horizon=cfg.horizon)
    raise ValueError(f"Unsupported model '{name}'. Expected 'lstm' or 'gru'.")


def iterate_minibatches(
    rng_key: jax.Array, inputs: jnp.ndarray, targets: jnp.ndarray, batch_size: int
) -> Iterable[Tuple[jnp.ndarray, jnp.ndarray]]:
    if inputs.shape[0] == 0:
        return []
    permutation = jax.random.permutation(rng_key, inputs.shape[0])
    for start in range(0, inputs.shape[0], batch_size):
        idx = permutation[start : start + batch_size]
        yield inputs[idx], targets[idx]


@dataclasses.dataclass
class TrainingResult:
    name: str
    history: Sequence[Dict[str, float]]
    params: Dict
    normalized_metrics: Dict[str, float]
    sales_metrics: Dict[str, float]
    breakdowns: Dict = None  # Metrics broken down by store, category, holiday, promotion


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mape(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = np.abs(targets) > epsilon
    if mask.sum() == 0:
        return 100.0
    percentage_errors = np.abs((targets[mask] - predictions[mask]) / targets[mask]) * 100
    return float(np.mean(percentage_errors))


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, and MAPE."""
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = compute_rmse(predictions, targets)
    mae = float(np.mean(np.abs(predictions - targets)))
    mape = compute_mape(predictions, targets)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def compute_metrics_by_store(
    predictions: np.ndarray,
    targets: np.ndarray,
    store_ids: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Metrics per store."""
    results: Dict[int, Dict[str, float]] = {}
    if store_ids is None:
        return results
    unique_stores = np.unique(store_ids)
    for store_id in unique_stores:
        mask = store_ids == store_id
        if mask.sum() == 0:
            continue
        store_preds = predictions[mask].flatten()
        store_targets = targets[mask].flatten()
        results[int(store_id)] = compute_all_metrics(store_preds, store_targets)
    return results


def compute_metrics_by_category(
    predictions: np.ndarray,
    targets: np.ndarray,
    family_ids: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Metrics per product family."""
    results: Dict[str, Dict[str, float]] = {}
    if family_ids is None:
        return results
    unique_families = np.unique(family_ids)
    for family in unique_families:
        mask = family_ids == family
        if mask.sum() == 0:
            continue
        fam_preds = predictions[mask].flatten()
        fam_targets = targets[mask].flatten()
        results[str(family)] = compute_all_metrics(fam_preds, fam_targets)
    return results


def compute_metrics_by_holiday(
    predictions: np.ndarray,
    targets: np.ndarray,
    is_holiday: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Metrics broken down by holiday flag."""
    results: Dict[str, Dict[str, float]] = {}
    if is_holiday is None:
        return results
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    holiday_flat = is_holiday.flatten().astype(bool)

    if holiday_flat.sum() > 0:
        results["holiday"] = compute_all_metrics(preds_flat[holiday_flat], targets_flat[holiday_flat])
    non_holiday = ~holiday_flat
    if non_holiday.sum() > 0:
        results["non_holiday"] = compute_all_metrics(preds_flat[non_holiday], targets_flat[non_holiday])
    return results


def compute_metrics_by_promotion(
    predictions: np.ndarray,
    targets: np.ndarray,
    has_promotion: np.ndarray,
    promo_threshold: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Metrics broken down by promotion flag."""
    results: Dict[str, Dict[str, float]] = {}
    if has_promotion is None:
        return results
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    promo_flat = has_promotion.flatten() > promo_threshold

    if promo_flat.sum() > 0:
        results["promotion"] = compute_all_metrics(preds_flat[promo_flat], targets_flat[promo_flat])
    no_promo = ~promo_flat
    if no_promo.sum() > 0:
        results["no_promotion"] = compute_all_metrics(preds_flat[no_promo], targets_flat[no_promo])
    return results


def create_run_directory(base_dir: Path, run_name: str | None) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    folder_name = run_name if run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / folder_name
    if run_dir.exists():
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = run_dir.parent / f"{run_dir.name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(run_dir: Path, cfg: TrainingConfig, results: Sequence[TrainingResult]) -> None:
    run_dir = Path(run_dir)
    serializable_cfg = dataclasses.asdict(cfg)
    serializable_cfg["data_dir"] = str(cfg.data_dir)
    serializable_cfg["families"] = list(cfg.families)
    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_cfg, f, indent=2)

    summary = {}
    for result in results:
        model_key = result.name.lower()
        artifact = {
            "normalized_metrics": result.normalized_metrics,
            "sales_metrics": result.sales_metrics,
            "history": list(result.history),
        }
        if result.breakdowns:
            artifact["breakdowns"] = {
                k: {str(k2): v2 for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in result.breakdowns.items()
            }
        with (run_dir / f"{model_key}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        
        # Save model parameters using pickle (JAX params are nested dicts)
        params_path = run_dir / f"{model_key}_params.pkl"
        with params_path.open("wb") as f:
            pickle.dump(result.params, f)
        
        summary[model_key] = {
            "normalized_metrics": result.normalized_metrics,
            "sales_metrics": result.sales_metrics,
        }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def train_model(
    name: str,
    model_def: nn.Module,
    dataset: DatasetSplits,
    cfg: TrainingConfig,
    rng: jax.Array,
) -> TrainingResult:
    rng, init_rng = jax.random.split(rng)
    sample_input = jnp.zeros((1, cfg.context_length, dataset.train_inputs.shape[-1]), dtype=jnp.float32)
    params = model_def.init(init_rng, sample_input)["params"]
    optimizer = optax.adam(cfg.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(p):
            preds = model_def.apply({"params": p}, batch_x)
            loss = jnp.mean((preds - batch_y) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(p, x, y):
        preds = model_def.apply({"params": p}, x)
        diff = preds - y
        mse = jnp.mean(diff**2)
        mae = jnp.mean(jnp.abs(diff))
        return mse, mae, preds

    history = []
    best_val = np.inf
    best_params = params

    for epoch in range(1, cfg.epochs + 1):
        rng, batch_rng = jax.random.split(rng)
        batch_losses = []
        for batch_x, batch_y in iterate_minibatches(batch_rng, dataset.train_inputs, dataset.train_targets, cfg.batch_size):
            params, opt_state, batch_loss = train_step(params, opt_state, batch_x, batch_y)
            batch_losses.append(batch_loss)
        train_loss = float(jnp.mean(jnp.stack(batch_losses))) if batch_losses else np.nan
        val_loss, val_mae, _ = eval_step(params, dataset.val_inputs, dataset.val_targets)
        metrics = {
            "epoch": epoch,
            "train_mse": float(train_loss),
            "val_mse": float(val_loss),
            "val_mae": float(val_mae),
        }
        history.append(metrics)
        if val_loss < best_val:
            best_val = float(val_loss)
            best_params = params
        _LOG.info(
            "[%s] Epoch %d/%d | train MSE=%.4f | val MSE=%.4f",
            name,
            epoch,
            cfg.epochs,
            metrics["train_mse"],
            metrics["val_mse"],
        )

    val_mse, val_mae, preds = eval_step(best_params, dataset.val_inputs, dataset.val_targets)
    preds_np = np.array(preds)
    targets_np = np.array(dataset.val_targets)
    sales_scaler = dataset.sales_scaler

    preds_sales = sales_scaler.denormalize_sales(preds_np)
    targets_sales = sales_scaler.denormalize_sales(targets_np)
    
    # Compute all metrics (normalized space)
    val_metrics_norm = compute_all_metrics(preds_np.flatten(), targets_np.flatten())
    
    # Compute all metrics (sales space)
    val_metrics_sales = compute_all_metrics(preds_sales.flatten(), targets_sales.flatten())
    
    # Compute breakdowns by store, category, holiday, promotion (validation only)
    breakdowns = {}
    if dataset.val_store_ids is not None and len(dataset.val_store_ids) > 0:
        breakdowns["by_store"] = compute_metrics_by_store(
            preds_np, targets_np, dataset.val_store_ids
        )
    if dataset.val_family_ids is not None and len(dataset.val_family_ids) > 0:
        breakdowns["by_category"] = compute_metrics_by_category(
            preds_np, targets_np, dataset.val_family_ids
        )
    if dataset.val_holiday_flags is not None and len(dataset.val_holiday_flags) > 0:
        breakdowns["by_holiday"] = compute_metrics_by_holiday(
            preds_np, targets_np, dataset.val_holiday_flags
        )
    if dataset.val_promo_flags is not None and len(dataset.val_promo_flags) > 0:
        breakdowns["by_promotion"] = compute_metrics_by_promotion(
            preds_np, targets_np, dataset.val_promo_flags
        )

    return TrainingResult(
        name=name,
        history=history,
        params=best_params,
        normalized_metrics={
            "val_mse": val_metrics_norm["mse"],
            "val_rmse": val_metrics_norm["rmse"],
            "val_mae": val_metrics_norm["mae"],
            "val_mape": val_metrics_norm["mape"],
        },
        sales_metrics={
            "val_mse_sales": val_metrics_sales["mse"],
            "val_rmse_sales": val_metrics_sales["rmse"],
            "val_mae_sales": val_metrics_sales["mae"],
            "val_mape_sales": val_metrics_sales["mape"],
        },
        breakdowns=breakdowns,
    )


def prepare_dataset(cfg: TrainingConfig) -> DatasetSplits:
    df, scaler = load_feature_frame(cfg)
    dataset = create_sequences(df, cfg, scaler)
    if dataset.train_inputs.shape[0] == 0 or dataset.val_inputs.shape[0] == 0:
        raise ValueError("Dataset split is empty — adjust filtering or date ranges.")
    return dataset


def train_models(
    cfg: TrainingConfig,
    model_names: Sequence[str] = ("lstm", "gru"),
    rng: Optional[jax.Array] = None,
) -> Tuple[Sequence[TrainingResult], DatasetSplits]:
    """
    Train one or more recurrent models and return their results plus the dataset.
    """
    dataset = prepare_dataset(cfg)
    if dataset.train_targets.size:
        effective_horizon = dataset.train_targets.shape[-1]
    elif dataset.val_targets.size:
        effective_horizon = dataset.val_targets.shape[-1]
    else:
        effective_horizon = cfg.horizon
    if effective_horizon != cfg.horizon:
        _LOG.warning(
            "Dataset horizon (%s) differs from config (%s). Updating config to match.",
            effective_horizon,
            cfg.horizon,
        )
        cfg = replace(cfg, horizon=int(effective_horizon))

    rng = rng or jax.random.PRNGKey(cfg.seed)
    results: List[TrainingResult] = []
    for name in model_names:
        rng, model_rng = jax.random.split(rng)
        model_def = build_model(name, cfg)
        result = train_model(name.upper(), model_def, dataset, cfg, model_rng)
        results.append(result)
    return results, dataset


def run_training_pipeline(
    cfg: TrainingConfig,
    output_dir: Path,
    run_name: Optional[str] = None,
    model_names: Sequence[str] = ("lstm", "gru"),
    rng: Optional[jax.Array] = None,
) -> Tuple[Path, Sequence[TrainingResult], DatasetSplits]:
    """
    Train models and persist metrics/parameters to a timestamped run directory.
    """
    results, dataset = train_models(cfg, model_names=model_names, rng=rng)
    run_dir = create_run_directory(output_dir, run_name)
    save_run_artifacts(run_dir, cfg, results)
    _LOG.info("Saved configuration, metrics, and parameters to %s", run_dir)
    return run_dir, results, dataset


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def load_run_config(run_dir: Path) -> TrainingConfig:
    """Load a serialized TrainingConfig from disk."""
    config_path = Path(run_dir) / "config.json"
    with config_path.open(encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["data_dir"] = Path(config_dict["data_dir"])
    config_dict["families"] = tuple(config_dict["families"])
    return TrainingConfig(**{k: v for k, v in config_dict.items() if k in TrainingConfig.__annotations__})


def load_model_params(run_dir: Path, model_name: str) -> Dict:
    """Load pickled model parameters."""
    params_path = Path(run_dir) / f"{model_name.lower()}_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters at {params_path}")
    with params_path.open("rb") as f:
        return pickle.load(f)


def run_inference(
    run_dir: Path,
    model_name: str,
    dataset: Optional[DatasetSplits] = None,
    cfg: Optional[TrainingConfig] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Load saved parameters and run predictions against the validation split.
    """
    jax.config.update("jax_platform_name", device)
    run_dir = Path(run_dir)
    if cfg is None:
        cfg = load_run_config(run_dir)
    if dataset is None:
        dataset = prepare_dataset(cfg)
    params = load_model_params(run_dir, model_name)
    model_def = build_model(model_name, cfg)

    @jax.jit
    def _predict_fn(p: Mapping[str, object], inputs: jnp.ndarray) -> jnp.ndarray:
        return model_def.apply({"params": p}, inputs)

    predictions = _predict_fn(params, dataset.val_inputs)
    preds_np = np.array(predictions)
    targets_np = np.array(dataset.val_targets)
    mse = float(np.mean((preds_np - targets_np) ** 2))
    mae = float(np.mean(np.abs(preds_np - targets_np)))
    _LOG.info(
        "Inference on %d windows with %s model complete | MSE=%.4f | MAE=%.4f",
        preds_np.shape[0],
        model_name.upper(),
        mse,
        mae,
    )
    return {"predictions": preds_np, "targets": targets_np, "mse": mse, "mae": mae}


# ---------------------------------------------------------------------------
# Plotting helpers (used by notebooks and docs)
# ---------------------------------------------------------------------------


def load_run_metrics(run_dir: Path) -> Dict[str, object]:
    """Load metric JSON blobs for both models."""
    run_dir = Path(run_dir)
    metrics: Dict[str, object] = {}
    for name in ("lstm", "gru"):
        metrics_path = run_dir / f"{name}_metrics.json"
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as f:
                metrics[name] = json.load(f)
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            metrics["config"] = json.load(f)
    return metrics


def plot_training_curves(metrics: Dict[str, object], output_dir: Path, model_name: Optional[str] = None) -> Path:
    """Plot training vs validation loss curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_to_plot = [model_name.lower()] if model_name else ["lstm", "gru"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

    for key in models_to_plot:
        if key not in metrics:
            continue
        history = metrics[key]["history"]
        epochs = [h["epoch"] for h in history]
        train_mse = [h["train_mse"] for h in history]
        val_mse = [h["val_mse"] for h in history]
        train_mae = [h.get("train_mae", 0) for h in history]
        val_mae = [h["val_mae"] for h in history]

        axes[0, 0].plot(epochs, train_mse, label=f"{key.upper()} Train", marker="o", alpha=0.7)
        axes[0, 0].plot(epochs, val_mse, label=f"{key.upper()} Val", marker="s", alpha=0.7)
        axes[0, 0].set_title("Mean Squared Error")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("MSE")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(epochs, train_mae, label=f"{key.upper()} Train", marker="o", alpha=0.7)
        axes[0, 1].plot(epochs, val_mae, label=f"{key.upper()} Val", marker="s", alpha=0.7)
        axes[0, 1].set_title("Mean Absolute Error")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].semilogy(epochs, train_mse, label=f"{key.upper()} Train", marker="o", alpha=0.7)
        axes[1, 0].semilogy(epochs, val_mse, label=f"{key.upper()} Val", marker="s", alpha=0.7)
        axes[1, 0].set_title("MSE (log scale)")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        loss_diff = [v - t for t, v in zip(train_mse, val_mse)]
        axes[1, 1].plot(epochs, loss_diff, label=f"{key.upper()} (Val-Train)", marker="o", alpha=0.7)
        axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.4)
        axes[1, 1].set_title("Validation - Training MSE")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Δ MSE")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_metrics_comparison(metrics: Dict[str, object], output_dir: Path) -> Path:
    """Bar chart comparing final normalized metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, val_mse, val_rmse, val_mae, val_mape = [], [], [], [], []
    for name in ("lstm", "gru"):
        if name in metrics:
            models.append(name.upper())
            norm = metrics[name]["normalized_metrics"]
            val_mse.append(norm.get("val_mse", 0))
            val_rmse.append(norm.get("val_rmse", 0))
            val_mae.append(norm.get("val_mae", 0))
            val_mape.append(norm.get("val_mape", 0))
    if not models:
        raise ValueError("No metrics available to plot.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Validation Metrics Comparison", fontsize=16, fontweight="bold")
    x = np.arange(len(models))
    width = 0.4
    for ax, values, title, ylabel in [
        (axes[0, 0], val_mse, "MSE", "MSE"),
        (axes[0, 1], val_rmse, "RMSE", "RMSE"),
        (axes[1, 0], val_mae, "MAE", "MAE"),
        (axes[1, 1], val_mape, "MAPE (%)", "MAPE (%)"),
    ]:
        ax.bar(x, values, width, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "final_metrics_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sales_metrics(metrics: Dict[str, object], output_dir: Path) -> Path:
    """Compare metrics computed in the original sales space."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, val_rmse_sales, val_mape_sales = [], [], []
    for name in ("lstm", "gru"):
        if name in metrics:
            models.append(name.upper())
            sales = metrics[name]["sales_metrics"]
            val_rmse_sales.append(sales.get("val_rmse_sales", 0))
            val_mape_sales.append(sales.get("val_mape_sales", 0))
    if not models:
        raise ValueError("No sales metrics available to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Validation Metrics (Sales Space)", fontsize=16, fontweight="bold")
    x = np.arange(len(models))
    axes[0].bar(x, val_rmse_sales, color="#0072B2", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("RMSE (sales units)")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x, val_mape_sales, color="#D55E00", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_title("MAPE (%) in sales space")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "sales_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_breakdown_bars(values: Mapping[str, float], title: str, ylabel: str, output_path: Path) -> None:
    """Utility to render a horizontal bar chart for breakdown metrics."""
    labels = list(values.keys())
    scores = list(values.values())
    plt.figure(figsize=(10, max(3, 0.4 * len(labels))))
    plt.barh(labels, scores, color="#009E73")
    plt.xlabel(ylabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_breakdowns(metrics: Dict[str, object], output_dir: Path, model_name: str) -> List[Path]:
    """Plot store/category/holiday/promotion breakdowns for a model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: List[Path] = []
    name = model_name.lower()
    if name not in metrics or "breakdowns" not in metrics[name]:
        return plots
    breakdowns = metrics[name]["breakdowns"]
    for key, filename in [
        ("by_store", f"{name}_metrics_by_store.png"),
        ("by_category", f"{name}_metrics_by_category.png"),
        ("by_holiday", f"{name}_metrics_by_holiday.png"),
        ("by_promotion", f"{name}_metrics_by_promotion.png"),
    ]:
        if key in breakdowns and breakdowns[key]:
            values = {str(k): v["rmse"] for k, v in breakdowns[key].items()}
            path = output_dir / filename
            plot_breakdown_bars(values, f"{model_name.upper()} {key.replace('_', ' ').title()}", "RMSE", path)
            plots.append(path)
    return plots


def plot_predictions_sample(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: DatasetSplits,
    output_path: Path,
    sample_index: int = 0,
) -> Path:
    """Plot a single validation horizon comparing predictions vs targets."""
    if predictions.size == 0:
        raise ValueError("No predictions available to plot.")
    plt.figure(figsize=(10, 4))
    plt.plot(targets[sample_index], label="Actual", marker="o")
    plt.plot(predictions[sample_index], label="Prediction", marker="x")
    plt.title("Validation horizon comparison")
    plt.xlabel("Forecast step")
    plt.ylabel("Normalized sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
