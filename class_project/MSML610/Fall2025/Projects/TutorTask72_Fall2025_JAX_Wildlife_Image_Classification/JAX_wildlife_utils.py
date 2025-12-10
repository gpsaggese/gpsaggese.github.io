"""
JAX Wildlife utilities.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import logging
import os
import random

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


_LOG = logging.getLogger(__name__)


# Data I/O

def list_images_with_labels(root: str) -> List[Tuple[str, int]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    items: List[Tuple[str, int]] = []
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((os.path.join(cdir, fn), cls_to_idx[c]))
    return items


def load_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(image_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_dataset(
    root: str,
    image_size: Tuple[int, int] = (128, 128),
    splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    shuffle: bool = True,
    seed: int = 42,
    limit_per_class: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1"
    items = list_images_with_labels(root)
    if limit_per_class is not None:
        grouped: Dict[int, List[str]] = {}
        for path, label in items:
            grouped.setdefault(label, []).append(path)
        items = []
        for label, paths in grouped.items():
            take = paths[:limit_per_class]
            items.extend([(p, label) for p in take])
    if shuffle:
        random.Random(seed).shuffle(items)
    X = np.stack([load_image(path, image_size) for path, _ in items])
    y = np.array([label for _, label in items], dtype=np.int32)
    n = len(X)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    X_splits = {"train": X[idx_train], "val": X[idx_val], "test": X[idx_test]}
    y_splits = {"train": y[idx_train], "val": y[idx_val], "test": y[idx_test]}
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    return X_splits, y_splits, class_names


def batch_iter(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        sel = idx[i : i + batch_size]
        yield X[sel], y[sel]


# Model + Training

class SimpleCNN(nn.Module):
    num_classes: int
    kernel_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (3, 3),
        (3, 3),
        (3, 3),
    )
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x, train: bool = True):
        ks1, ks2, ks3 = self.kernel_sizes
        x = nn.Conv(features=32, kernel_size=ks1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(features=64, kernel_size=ks2)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.Conv(features=128, kernel_size=ks3)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.num_classes)(x)
        return x


@dataclass
class TrainConfig:
    image_size: Tuple[int, int] = (128, 128)
    num_classes: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 5
    seed: int = 0
    conv_kernel_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (3, 3),
        (3, 3),
        (3, 3),
    )
    dropout_rate: float = 0.5


def create_train_state(rng: jax.Array, config: TrainConfig) -> train_state.TrainState:
    model = SimpleCNN(
        num_classes=config.num_classes,
        kernel_sizes=config.conv_kernel_sizes,
        dropout_rate=config.dropout_rate,
    )
    rng, init_dropout = jax.random.split(rng)
    params = model.init(
        {"params": rng, "dropout": init_dropout},
        jnp.ones((1, config.image_size[0], config.image_size[1], 3), dtype=jnp.float32),
        train=True,
    )["params"]
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    onehot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return optax.softmax_cross_entropy(logits, onehot).mean()


@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], rng: jax.Array):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["X"], train=True, rngs={"dropout": rng})
        loss = cross_entropy_loss(logits, batch["y"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean((preds == batch["y"]).astype(jnp.float32))
    return state, loss, acc


@jax.jit
def eval_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    logits = state.apply_fn({"params": state.params}, batch["X"], train=False)
    preds = jnp.argmax(logits, axis=-1)
    return preds


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Tuple[train_state.TrainState, Dict[str, List[float]]]:
    rng = jax.random.PRNGKey(config.seed)
    state = create_train_state(rng, config)
    history = {"loss": [], "acc": [], "val_acc": []}
    for epoch in range(1, config.num_epochs + 1):
        losses: List[float] = []
        accs: List[float] = []
        for Xb, yb in batch_iter(X_train, y_train, config.batch_size, shuffle=True, seed=epoch):
            rng, sub = jax.random.split(rng)
            batch = {"X": jnp.array(Xb), "y": jnp.array(yb)}
            state, loss, acc = train_step(state, batch, sub)
            losses.append(float(loss))
            accs.append(float(acc))
        preds_list: List[np.ndarray] = []
        for Xb, yb in batch_iter(X_val, y_val, config.batch_size, shuffle=False):
            batch = {"X": jnp.array(Xb), "y": jnp.array(yb)}
            preds = eval_step(state, batch)
            preds_list.append(np.array(preds))
        val_preds = np.concatenate(preds_list) if preds_list else np.array([], dtype=np.int32)
        val_acc = float(accuracy_score(y_val[: len(val_preds)], val_preds)) if len(val_preds) else 0.0
        _LOG.info(
            "Epoch %d/%d loss=%.4f acc=%.4f val_acc=%.4f",
            epoch,
            config.num_epochs,
            np.mean(losses) if losses else 0.0,
            np.mean(accs) if accs else 0.0,
            val_acc,
        )
        history["loss"].append(np.mean(losses) if losses else 0.0)
        history["acc"].append(np.mean(accs) if accs else 0.0)
        history["val_acc"].append(val_acc)
    return state, history


def evaluate(
    state: train_state.TrainState,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
) -> Dict[str, object]:
    preds_list: List[np.ndarray] = []
    for Xb, yb in batch_iter(X, y, 256, shuffle=False):
        batch = {"X": jnp.array(Xb), "y": jnp.array(yb)}
        preds = eval_step(state, batch)
        preds_list.append(np.array(preds))
    y_pred = np.concatenate(preds_list) if preds_list else np.array([], dtype=np.int32)
    acc = accuracy_score(y, y_pred) if len(y_pred) else 0.0
    prec = precision_score(y, y_pred, average="macro", zero_division=0) if len(y_pred) else 0.0
    rec = recall_score(y, y_pred, average="macro", zero_division=0) if len(y_pred) else 0.0
    cm = confusion_matrix(y, y_pred) if len(y_pred) else np.zeros((len(class_names), len(class_names)), dtype=int)
    return {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm, "y_pred": y_pred}


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], figsize=(6, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    return fig


def sample_misclassifications(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.where(y_true != y_pred)[0]
    if len(idx) == 0:
        return np.empty((0, *X.shape[1:])), np.array([]), np.array([])
    take = idx[:k]
    return X[take], y_true[take], y_pred[take]
