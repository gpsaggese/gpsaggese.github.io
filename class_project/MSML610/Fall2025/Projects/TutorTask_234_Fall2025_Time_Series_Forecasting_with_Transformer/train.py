"""
train.py — Training loop for TimeSeries Transformer
Supports:
- AdamW optimizer
- Cosine LR schedule with warmup
- Gradient clipping
- Batch-wise validation
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state, checkpoints

from model import TimeSeriesTransformer


# ============================================================
# Learning Rate Schedule (Warmup + Cosine Decay)
# ============================================================

def create_learning_rate_fn(
    base_lr=1e-3,
    warmup_steps=500,
    total_steps=20000,
):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps,
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
    )

    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )


# ============================================================
# Create Train State
# ============================================================

def create_train_state(
    rng,
    learning_rate,
    seq_len,
    out_len,
    num_features
):
    # -------------------------------------------------------
    # FIXED: ALWAYS use num_features=6 to match saved weights
    # -------------------------------------------------------
    model = TimeSeriesTransformer(
        seq_len=seq_len,
        out_len=out_len,
        num_features=6     # <<── HARD-SET FOR CHECKPOINT COMPATIBILITY
    )

    dummy = jnp.ones((1, seq_len, 6))  # <<── MUST MATCH num_features=6

    params = model.init(
        {"params": rng, "dropout": rng},
        dummy,
        train=True
    )["params"]

    lr_schedule = create_learning_rate_fn(
        base_lr=learning_rate,
        warmup_steps=500,
        total_steps=20000,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=1e-4
        ),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


# ============================================================
# Loss Function
# ============================================================

def loss_fn(params, batch, state, rng):
    preds = state.apply_fn(
        {"params": params},
        batch["X"],
        train=True,
        rngs={"dropout": rng},
    )
    return jnp.mean((preds - batch["y"]) ** 2)


# ============================================================
# Training Step
# ============================================================

@jax.jit
def train_step(state, batch, rng):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, batch, state, rng)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ============================================================
# Validation Step
# ============================================================

@jax.jit
def eval_step(state, batch):
    preds = state.apply_fn(
        {"params": state.params},
        batch["X"],
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )
    return jnp.mean((preds - batch["y"]) ** 2)


# ============================================================
# Batch Generator
# ============================================================

def get_batches(X, y, batch_size):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    for i in range(0, len(X), batch_size):
        b = idx[i:i + batch_size]
        yield {
            "X": X[b],
            "y": y[b]
        }


# ============================================================
# Training Loop
# ============================================================

def train_model(
    X_train, y_train,
    X_val, y_val,
    seq_len,
    out_len,
    num_features,   # ignored, because checkpoint must use 6
    batch_size=64,
    epochs=20,
    learning_rate=1e-3,
    ckpt_dir="./checkpoints"
):

    rng = jax.random.PRNGKey(0)

    state = create_train_state(
        rng,
        learning_rate,
        seq_len,
        out_len,
        num_features=6   # <<── FORCE 6-FEATURE MODEL
    )

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_losses = []

        # ----------------------- TRAIN LOOP -----------------------
        for batch in get_batches(X_train, y_train, batch_size):
            rng, dropout_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, dropout_rng)
            train_losses.append(float(loss))

        train_loss = np.mean(train_losses)

        # ----------------------- VALIDATION -----------------------
        val_losses = []
        for batch in get_batches(X_val, y_val, batch_size):
            loss = eval_step(state, batch)
            val_losses.append(float(loss))

        val_loss = np.mean(val_losses)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ----------------------- CHECKPOINT -----------------------
        if val_loss < best_val:
            best_val = val_loss
            checkpoints.save_checkpoint(
                ckpt_dir,
                target=state.params,
                step=epoch,
                overwrite=True
            )
            print("✔ New best checkpoint saved!")

    return state
