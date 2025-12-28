# JAX_wildlife.example – Complete Walkthrough
**Project:** TutorTask72 – Wildlife Image Classification with JAX/Flax  
**Notebook:** `JAX_wildlife.example.ipynb`


---

## 0. Prerequisites
```bash
cd TutorTask72_Fall2025_JAX_Wildlife_Image_Classification
docker build -t msml610/tutortask72_jax .
docker run --rm -it -p 8888:8888 -v "${PWD}:/data" msml610/tutortask72_jax \
  bash -lc "cd /data && jupyter-notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"
```
Then open `http://localhost:8888` and run `JAX_wildlife.example.ipynb` via “Restart & Run All.”

---

## 1. Notebook Goals
1. Load the full Animals-10 dataset.
2. Train the configurable `SimpleCNN` on train/validation splits.
3. Evaluate on the held-out test split and persist all diagnostics.
4. Run a hyper-parameter sweep on a reduced dataset for rapid comparison.
5. Save visualizations under `outputs/evaluation/` and `outputs/hparam/`.

Workflow diagram:
```
data/animals10/ -> load_dataset -> train() -> evaluate() -> outputs/evaluation/
                                        \
                                         `-> hyper-parameter sweep -> outputs/hparam/
```

---

## 2. Step-by-Step Narrative

### Step 1 – Imports & Directories
```python
import logging, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from JAX_wildlife_utils import (
    load_dataset, TrainConfig, train, evaluate,
    plot_confusion_matrix, sample_misclassifications
)

DATA_DIR = './data/animals10'
IMAGE_SIZE = (128, 128)
OUTPUT_DIR = Path('outputs')
EVAL_DIR = OUTPUT_DIR / 'evaluation'
HPARAM_DIR = OUTPUT_DIR / 'hparam'
for path in (OUTPUT_DIR, EVAL_DIR, HPARAM_DIR):
    path.mkdir(exist_ok=True)
```
**Intent:** Centralize configuration and ensure output folders exist. Using `Path` avoids platform-specific path issues.

### Step 2 – Load Dataset
```python
LIMIT_PER_CLASS = None      
TRAIN_EPOCHS = 3            
Xs, ys, class_names = load_dataset(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    splits=(0.7, 0.15, 0.15),
    limit_per_class=LIMIT_PER_CLASS
)
```

### Step 3 – Train
```python
config = TrainConfig(
    image_size=IMAGE_SIZE,
    num_classes=len(class_names),
    num_epochs=TRAIN_EPOCHS,
    batch_size=64,
    learning_rate=1e-3
)
state, history = train(Xs['train'], ys['train'], Xs['val'], ys['val'], config)
```
- Records `history['loss']`, `history['acc']`, and `history['val_acc']` per epoch.

### Step 4 – Evaluate & Persist Metrics
```python
metrics = evaluate(state, Xs['test'], ys['test'], class_names)
fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
fig.savefig(EVAL_DIR / 'confusion_matrix.png', bbox_inches='tight')
```
Outputs:
- `metrics['accuracy']`, `metrics['precision']`, `metrics['recall']`, `metrics['confusion_matrix']`, `metrics['y_pred']`
- `outputs/evaluation/confusion_matrix.png`

### Step 5 – Metric Bar Chart
```python
metric_values = {k: metrics[k] for k in ['accuracy', 'precision', 'recall']}
fig, ax = plt.subplots(figsize=(4, 3))
bars = ax.bar(metric_values.keys(), metric_values.values(),
              color=['#4caf50', '#2196f3', '#ff9800'])
...
fig.savefig(EVAL_DIR / 'test_metrics.png', bbox_inches='tight')
```

### Step 6 – Hyper-Parameter Sweep (Reduced Dataset)
```python
Xs_tune, ys_tune, _ = load_dataset(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    splits=(0.7, 0.15, 0.15),
    limit_per_class=40
)
```
We define four experiments:
1. **baseline** – kernels `(3,3)` and dropout `0.5`
2. **wide-kernel** – first kernel `(5,5)`
3. **lower-dropout** – dropout `0.3`
4. **low-lr** – learning rate `5e-4`
Each configuration trains for 2 epochs, logs validation accuracy/precision/recall, and appends results to `tuning_results`.

### Step 7 – Plot Tuning Accuracy
```python
names = [cfg['name'] for cfg in tuning_grid]
val_acc = [result['val_accuracy'] for result in tuning_results]
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(names, val_acc, marker='o')
fig.savefig(HPARAM_DIR / 'hparam_tuning.png', bbox_inches='tight')
```
This PNG shows how accuracy shifts across hyper-parameter choices.

### Step 8 – Confusion Matrices per Variant
For more granular insight, each configuration is retrained (on the subset) and `confusion_<experiment>.png` is saved under `outputs/hparam/`.

### Step 9 – Misclassifications & Correct Examples
```python
images, y_true, y_pred = sample_misclassifications(...)
fig.savefig(EVAL_DIR / 'misclassifications.png', ...)

correct_idx = np.where(metrics['y_pred'] == ys['test'])[0]
fig.savefig(EVAL_DIR / 'correct_examples.png', ...)
```

---

## 3. Outputs Snapshot
```
outputs/
├── evaluation/
│   ├── confusion_matrix.png
│   ├── test_metrics.png
│   ├── misclassifications.png
│   └── correct_examples.png
└── hparam/
    ├── hparam_tuning.png
    ├── confusion_baseline.png
    ├── confusion_wide-kernel.png
    ├── confusion_lower-dropout.png
    └── confusion_low-lr.png
```

---

## 5. Sample Script (Outside Notebook)
```python
from pathlib import Path
from JAX_wildlife_utils import load_dataset, TrainConfig, train, evaluate, plot_confusion_matrix

OUTPUT_DIR = Path('outputs/eval_script')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Xs, ys, class_names = load_dataset('./data/animals10', image_size=(128,128))
config = TrainConfig(image_size=(128,128), num_classes=len(class_names), num_epochs=3)
state, _ = train(Xs['train'], ys['train'], Xs['val'], ys['val'], config)
metrics = evaluate(state, Xs['test'], ys['test'], class_names)
fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
fig.savefig(OUTPUT_DIR / 'confusion.png', bbox_inches='tight')
```
This script version is useful for automation or CI jobs if you need to run outside Jupyter.

---

## 6. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Notebook takes too long | Full dataset + 3 epochs on CPU | Use `LIMIT_PER_CLASS = 40`. |
| `MemoryError` during load | Insufficient RAM | Reduce `IMAGE_SIZE` or use `limit_per_class`. |
| Confusion matrices empty | No predictions recorded | Ensure test split isn’t empty (check `LIMIT_PER_CLASS`). |
| Plots missing | Folders not created | Verify the first cell (Path setup) ran successfully. |
| GPU not detected | CUDA/JAX mismatch | Run with CPU; document that GPU support requires installing the CUDA plugin wheels. |

---

## 7. Future Improvements
- Integrate data augmentation (random flips/crops) inside `load_dataset` or via TensorFlow Datasets.
- Swap `SimpleCNN` with a pre-trained vision backbone if GPU time is available.
- Add automated LR scheduling / early stopping hooks in the training loop.
- Export metrics to JSON for easier comparison across multiple runs.

---

