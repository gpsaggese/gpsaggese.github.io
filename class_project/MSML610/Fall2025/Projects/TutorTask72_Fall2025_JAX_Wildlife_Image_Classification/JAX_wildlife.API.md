# JAX_wildlife.API
**Project:** TutorTask72 - Wildlife Image Classification with JAX/Flax  
**Author:** Ravi Vignesh LNU (UID 121322302)

This document describes the API of the tool itself. It focuses on the native JAX programming surface we rely on and the thin wrapper layer in `JAX_wildlife_utils.py` that standardizes how callers load data, configure the model, train, and evaluate. External data-provider APIs are intentionally out of scope.

---

## 1. Native JAX/Flax Surface (What We Build On)
- `flax.linen.Module`: base class for neural network components. Our `SimpleCNN` subclasses it and implements `__call__(self, x, train: bool) -> logits`.
- `flax.training.train_state.TrainState`: lightweight container for `apply_fn`, `params`, and optimizer state; created in `create_train_state`.
- `optax.adam(learning_rate)`: returns a gradient transformation used by `TrainState`.
- `jax.jit` and `jax.value_and_grad`: compile `train_step`/`eval_step` for speed and differentiate the loss.
- `jax.random.PRNGKey` and `jax.random.split`: manage RNG streams for parameter init and dropout.

These are the "native API" elements users should expect our wrapper to rely on; the wrapper keeps direct JAX usage contained so integrators only need to depend on the interface described below.

---

## 2. Core Objects (Runtime-Independent)
```python
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Protocol
import numpy as np
import jax.numpy as jnp
from flax.training import train_state

@dataclass
class TrainConfig:
    image_size: Tuple[int, int] = (128, 128)
    num_classes: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 5
    seed: int = 0
    conv_kernel_sizes: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((3, 3), (3, 3), (3, 3))
    dropout_rate: float = 0.5

class ClassifierAPI(Protocol):
    def create_state(self, rng: jnp.ndarray, config: TrainConfig) -> train_state.TrainState: ...
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: TrainConfig) -> Tuple[train_state.TrainState, Dict[str, List[float]]]: ...
    def predict(self, state: train_state.TrainState, X_batch: np.ndarray) -> np.ndarray: ...
```
- `TrainConfig` is the configuration object; no file paths or side effects.
- `ClassifierAPI` describes the minimal interface that any implementation (ours or yours) should satisfy to plug into notebooks or scripts without bringing in a specific runtime.

---

## 3. Wrapper Layer (What `JAX_wildlife_utils.py` Provides)
| Symbol | Responsibility |
|--------|----------------|
| `list_images_with_labels` | Enumerate class folders and return `(path, class_index)` tuples with stable ordering. |
| `load_image` | Open image with Pillow, convert to RGB, resize bilinearly, normalize to `[0, 1]`. |
| `load_dataset` | Deterministic ingestion pipeline returning `(X_splits, y_splits, class_names)`; honors per-class caps early for quick demos. |
| `batch_iter` | Yield mini-batches without copying full arrays. |
| `SimpleCNN` | `nn.Module` with `__call__(x, train=True)`: three Conv -> ReLU -> MaxPool blocks, then Dense -> Dropout -> Dense. Kernel sizes/dropout come from `TrainConfig`. |
| `create_train_state` | Initializes `SimpleCNN`, params, and Optax Adam optimizer. |
| `train_step` | JITed forward/backward pass with dropout RNG. |
| `eval_step` | JITed inference for a batch. |
| `train` | Epoch loop that logs loss/acc and validation accuracy. |
| `evaluate` | Computes accuracy, macro precision/recall, confusion matrix, and raw `y_pred`. |
| `plot_confusion_matrix` | Returns a labeled Matplotlib figure; caller decides whether to display/save. |
| `sample_misclassifications` | First `k` mistakes as `(images, y_true_subset, y_pred_subset)` or empty arrays if none. |

---

## 4. Mapping to the Native API
- `SimpleCNN` is the only `flax.linen.Module` exposed; callers do not manipulate Flax internals directly.
- `create_train_state` hides `train_state.TrainState` construction and `optax.adam` wiring.
- `train_step` and `eval_step` are the only `jax.jit`/`jax.value_and_grad` touchpoints. 

---

## 5. Minimal Usage (shown in `JAX_wildlife.API.ipynb`)
```python
from JAX_wildlife_utils import load_dataset, TrainConfig, create_train_state, train, evaluate

DATA_DIR = "./data/animals10"
IMAGE_SIZE = (128, 128)

from JAX_wildlife_utils import SimpleCNN
import jax, jax.numpy as jnp
rng = jax.random.PRNGKey(0)
model = SimpleCNN(num_classes=10)
params = model.init({"params": rng, "dropout": rng}, jnp.ones((1, *IMAGE_SIZE, 3)), train=True)["params"]
logits = model.apply({"params": params}, jnp.ones((1, *IMAGE_SIZE, 3)), train=False)

# Wrapper workflow
Xs, ys, class_names = load_dataset(DATA_DIR, image_size=IMAGE_SIZE, splits=(0.7, 0.15, 0.15), limit_per_class=20)
config = TrainConfig(image_size=IMAGE_SIZE, num_classes=len(class_names), num_epochs=1, batch_size=32)
state, history = train(Xs["train"], ys["train"], Xs["val"], ys["val"], config)
metrics = evaluate(state, Xs["test"], ys["test"], class_names)
```
- First cell proves the native API (Flax module) works in isolation.
- Subsequent cells exercise the wrapper flow end to end with small inputs to keep the notebook fast.

---

## 6. API vs. Example
- `JAX_wildlife.API.*` (this file and `JAX_wildlife.API.ipynb`) define and demonstrate the core interfaces only.
- `JAX_wildlife.example.*` provides a runnable reference implementation that expands on this API with full training runs, output persistence, and hyper-parameter sweeps. It can be swapped out if a different backend is desired as long as it respects the API signatures above.

---

## 7. Extending or Replacing the Implementation
- Swap `SimpleCNN` for another `nn.Module` by keeping the same `__call__` signature and returning logits of shape `(batch, num_classes)`.
- Implement your own `ClassifierAPI` that satisfies `create_state`/`train`/`predict` to integrate alternative optimizers or architectures without changing the API notebook.
- For data ingestion, you can provide another `load_dataset` implementation as long as it returns the same `(X_splits, y_splits, class_names)` structure.
