"""
transformer_utils.py

Reusable utilities for:
- Data preparation
- Transformer model definition
- Training
- Evaluation
"""

# ===================== DATA =====================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ticker_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").set_index("Date")
    return df


def create_price_targets(df, horizon=5):
    close = df["Close"].values
    y = []
    for i in range(len(close) - horizon):
        y.append(close[i + 1 : i + 1 + horizon])
    return np.array(y)


def create_windows(X, y, seq_len):
    X_out, y_out = [], []
    for i in range(len(X) - seq_len):
        X_out.append(X[i : i + seq_len])
        y_out.append(y[i + seq_len - 1])
    return np.array(X_out), np.array(y_out)


def prepare_dataset(path, seq_len=60, horizon=5):
    df = load_ticker_csv(path)

    y_raw = create_price_targets(df, horizon)
    X_raw = df.iloc[:-horizon].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    close = df["Close"].values
    close_mean, close_std = close.mean(), close.std()
    y_scaled = (y_raw - close_mean) / close_std

    X, y = create_windows(X_scaled, y_scaled, seq_len)

    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
        scaler_X,
        close_mean,
        close_std,
    )


# ===================== MODEL =====================

import jax
import jax.numpy as jnp
import flax.linen as nn


class PositionalEncoding(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x):
        pos = self.param(
            "pos",
            nn.initializers.normal(stddev=0.01),
            (5000, self.d_model),
        )
        return x + pos[: x.shape[1]]


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, train=True):
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=0.0,
        )
        x2 = attn(x, deterministic=True)
        x = nn.LayerNorm()(x + x2)

        x2 = nn.Dense(self.mlp_dim)(x)
        x2 = nn.gelu(x2)
        x2 = nn.Dense(self.d_model)(x2)

        return nn.LayerNorm()(x + x2)


class TimeSeriesTransformer(nn.Module):
    seq_len: int = 60
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = 256
    out_len: int = 5
    num_features: int = 6

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(self.d_model)(x)
        x = PositionalEncoding(self.d_model)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
            )(x, train=train)

        return nn.Dense(self.out_len)(x[:, -1])


# ===================== TRAINING =====================

import optax
from flax.training import train_state, checkpoints


def create_learning_rate_fn(base_lr=1e-3, warmup_steps=500, total_steps=20000):
    warmup = optax.linear_schedule(0.0, base_lr, warmup_steps)
    cosine = optax.cosine_decay_schedule(base_lr, total_steps - warmup_steps)
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def create_train_state(rng, seq_len, out_len):
    model = TimeSeriesTransformer(seq_len=seq_len, out_len=out_len)
    dummy = jnp.ones((1, seq_len, 6))

    params = model.init(
        {"params": rng, "dropout": rng},
        dummy,
        train=True,
    )["params"]

    tx = optax.adamw(create_learning_rate_fn())
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def loss_fn(params, state, batch, rng):
    preds = state.apply_fn(
        {"params": params},
        batch["X"],
        train=True,
        rngs={"dropout": rng},
    )
    return jnp.mean((preds - batch["y"]) ** 2)


@jax.jit
def train_step(state, batch, rng):
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, state, batch, rng
    )
    return state.apply_gradients(grads=grads), loss


def get_batches(X, y, batch_size):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        b = idx[i : i + batch_size]
        yield {"X": X[b], "y": y[b]}


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    seq_len,
    out_len,
    epochs=20,
    batch_size=64,
    ckpt_dir="./checkpoints",
):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, seq_len, out_len)

    best_val = float("inf")

    for epoch in range(epochs):
        for batch in get_batches(X_train, y_train, batch_size):
            rng, sub = jax.random.split(rng)
            state, _ = train_step(state, batch, sub)

        val_loss = np.mean(
            [
                float(
                    jnp.mean(
                        (state.apply_fn({"params": state.params}, b["X"]) - b["y"]) ** 2
                    )
                )
                for b in get_batches(X_val, y_val, batch_size)
            ]
        )

        if val_loss < best_val:
            best_val = val_loss
            checkpoints.save_checkpoint(
                ckpt_dir, state.params, epoch, overwrite=True
            )

    return state




# ===================== EVALUATION =====================

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def predict(model, params, X):
    return np.array(
        model.apply(
            {"params": params},
            jnp.array(X),
            train=False,
        )
    )


def compute_metrics(y_true, y_pred):
    return {
        f"MAE_h{i+1}": float(np.mean(np.abs(y_true[:, i] - y_pred[:, i])))
        for i in range(y_true.shape[1])
    }


def plot_horizon(y_true, y_pred, horizon=1, n=200):
    h = horizon - 1
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:n, h], label="Actual")
    plt.plot(y_pred[:n, h], label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()

def predict_next_5_days(model, params, last_window_scaled, scaler_y):
    """
    Predict the next 5 timesteps in real price space.
    """
    import jax.numpy as jnp
    import numpy as np
    import jax

    window = jnp.array(last_window_scaled)[None, :, :]

    y_scaled = model.apply(
        {"params": params},
        window,
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )[0]

    y_real = scaler_y.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).flatten()

    return y_real, np.array(y_scaled)



# ============================================================
# MODEL LOADING (for evaluation & inference)
# ============================================================

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np

def load_model(ckpt_dir, seq_len, out_len, num_features):
    """
    Load a trained TimeSeriesTransformer and restore parameters.
    """

    model = TimeSeriesTransformer(
        seq_len=seq_len,
        out_len=out_len,
        num_features=num_features,
    )

    dummy = jnp.ones((1, seq_len, num_features))

    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        dummy,
        train=False,
    )["params"]

    params = checkpoints.restore_checkpoint(
        ckpt_dir,
        target=params,
    )

    return model, params

def load_data(path, config):
    """
    High-level data loading wrapper for API usage.
    """
    return prepare_dataset(
        path,
        seq_len=config["seq_len"],
        horizon=config["horizon"],
    )[:6]  # only return X/y splits

def train_and_evaluate(X_train, y_train, X_val, y_val, config):
    """
    High-level API wrapper for training a transformer model.

    Returns:
        model: trained TimeSeriesTransformer
        params: trained parameters
    """
    seq_len = config["seq_len"]
    out_len = config["horizon"]
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 64)
    ckpt_dir = config.get("ckpt_dir", "./checkpoints")

    # Train
    train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        seq_len=seq_len,
        out_len=out_len,
        epochs=epochs,
        batch_size=batch_size,
        ckpt_dir=ckpt_dir,
    )

    # Load best model
    model, params = load_model(
        ckpt_dir=ckpt_dir,
        seq_len=seq_len,
        out_len=out_len,
        num_features=X_train.shape[-1],
    )

    return model, params

