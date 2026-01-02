# AutoKeras API (Tool Overview)

This document explains **AutoKeras** as a tool for image classification, focusing on what it is, how to use it, and its core functionality.  
It is **tool-only** (not tied to any specific project dataset or pipeline).

---

## 1. What is AutoKeras?

**AutoKeras** is an AutoML library built on top of TensorFlow/Keras. For image tasks, it can automatically search for a good neural network architecture and training pipeline, so you don’t have to manually design a CNN from scratch.

For image classification, the main high-level tool is:

- **`ak.ImageClassifier`** — runs a small Neural Architecture Search (NAS) / AutoML search over candidate models and picks the best one based on validation performance.

---

## 2. Key AutoKeras Objects and Methods

### `ak.ImageClassifier(...)`
Creates an AutoKeras image classifier that will:
- try multiple candidate model architectures (`max_trials`)
- train each candidate for a limited number of epochs
- select the best model using validation metrics

Common parameters:
- **`max_trials`**: how many candidate models to try  
- **`overwrite`**: whether to overwrite previous search results

### `fit(...)`
Trains the search:
- AutoKeras runs training for each trial and tracks validation performance.

### `evaluate(...)`
Evaluates the best found pipeline/model on a held-out test set.

### `export_model()`
Exports the best discovered model as a standard Keras model (`tf.keras.Model`) so you can:
- view `model.summary()`
- save it to disk
- deploy it like any other Keras model

### `model.save(...)`
Saves the exported model in a portable `.keras` format.

---

## 3. Minimal Usage Example (Image Classification)

Below is a small end-to-end example demonstrating the AutoKeras workflow:

```python
import tensorflow as tf
import autokeras as ak

# 1) Load data (example: CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2) Keep the demo small (so it runs quickly even on CPU)
x_train_small, y_train_small = x_train[:500], y_train[:500]
x_val_small, y_val_small = x_train[500:600], y_train[500:600]
x_test_small, y_test_small = x_test[:200], y_test[:200]

# 3) Create the AutoKeras classifier
clf = ak.ImageClassifier(max_trials=2, overwrite=True)

# 4) Run a small search + training
clf.fit(
    x_train_small, y_train_small,
    validation_data=(x_val_small, y_val_small),
    epochs=2
)

# 5) Evaluate
test_loss, test_acc = clf.evaluate(x_test_small, y_test_small, verbose=0)
print("Demo test accuracy:", test_acc)

# 6) Export and save the best model
best_model = clf.export_model()
best_model.summary()
best_model.save("autokeras_image_classifier_demo.keras")
```

Notes:
- If you see Available GPUs: [] or CUDA warnings, that just means the environment is CPU-only. AutoKeras still works on CPU.
- In a demo notebook, max_trials and epochs are intentionally kept small to minimize runtime. Accuracy is not the goal in a tiny demo—showing the workflow is.

---

## 4. Common AutoKeras Workflow Summary
Typical workflow for AutoKeras image classification:
1. Prepare data (NumPy arrays are the simplest for AutoKeras demos)
2. Create ak.ImageClassifier(max_trials=...)
3. Fit with clf.fit(...) and validation data
4. Evaluate using clf.evaluate(...)
5. Export best model via clf.export_model()
6. Save the Keras model for reuse/deployment
This is the core set of AutoKeras functionalities used in most practical image-classification tasks.