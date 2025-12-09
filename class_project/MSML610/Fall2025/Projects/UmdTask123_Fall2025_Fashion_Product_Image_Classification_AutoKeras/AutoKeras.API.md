# AutoKeras API – Fashion Product Image Classification

This notebook does not aim to train the strongest model.  
Instead, it defines the **API layer** used across the entire project — a simple, reusable interface for:

- loading the Fashion Product Images dataset  
- building the baseline CNN  
- initializing the AutoKeras ImageClassifier  

By centralizing these components, the main experiment notebook stays clean and focused, while this notebook documents how the API works.

---

## 1. API Files Used in This Notebook

### utils_data_io.py — Data Loading Helpers

**1. `tsv_to_tfds(tsv_path, num_classes)`**  
- Reads a TSV file containing `image_path` and `label_idx`.  
- Loads and decodes each image.  
- Resizes everything to **224×224**.  
- Returns a `tf.data.Dataset` of `(image, label)`.

**2. `ds_to_numpy(ds, max_samples)`**  
- Iterates through a dataset and collects up to `max_samples`.  
- Returns `(X, y)` as NumPy arrays.  
- Useful because AutoKeras prefers NumPy input.

---

## utils_model.py — Model Builder Helpers

**1. `make_baseline_cnn(input_shape, num_classes)`**  
- Builds a simple CNN with three Conv+Pool blocks, followed by Flatten → Dense → Dropout → Softmax.  
- Compiles with Adam and sparse categorical cross-entropy.  
- Serves as the baseline for comparison.

**2. `make_autokeras_image_classifier(num_classes, max_trials)`**  
- Returns a configured AutoKeras ImageClassifier.  
- Keeps the search budget (`max_trials`) consistent across notebooks.

---

## 2. What This Notebook Demonstrates

### TensorFlow Environment Check
The notebook confirms TensorFlow installation and device availability.  
In Docker, expected output is:

TensorFlow version: 2.20.0
Available GPUs: [] 

CPU-only Docker is normal here.

---

### Loading a Small Demo Batch Using the API

To keep this API notebook lightweight, only **64 samples** are loaded.  
This validates that:

- TSV parsing works  
- image decoding and resizing work  
- labels map correctly  

Shapes:

X_small → (64, 224, 224, 3)
y_small → (64,)


A small image grid is plotted to visually confirm everything is correct.

---

### Building the Baseline CNN

The baseline model is created with one function call:

```python
baseline_model = make_baseline_cnn(
    input_shape=X_small.shape[1:], 
    num_classes=NUM_CLASSES,
)

A quick 1-epoch training pass verifies:

- model compiles successfully
- pipeline runs end-to-end
- training progresses without errors

Full training is done in AutoKeras.example.ipynb.

### Initializing the AutoKeras Classifier

```python
    clf = make_autokeras_image_classifier(
    num_classes=NUM_CLASSES,
    max_trials=2
)

This confirms:

- AutoKeras is installed
- the wrapper function returns a valid classifier

A full neural architecture search is not performed here because CPU-only Docker is slow.
A usage template is provided:

```python
clf.fit(
    train_ds.batch(32),
    epochs=5,
    validation_data=val_ds.batch(32),
)


The heavy AutoKeras search runs on Colab with GPU.

## 3. Design Decisions
### 1. Clear Separation of Concerns

- Data logic in utils_data_io.py
- Model logic in utils_model.py
- API demonstration here

### 2. Docker-Friendly Workflow

- Very small data subset
- Minimal runtime
- Ensures anyone can run everything quickly and reliably

### 3. Minimal but Complete API Surface

The entire project pipeline relies on just four functions:
- tsv_to_tfds
- ds_to_numpy
- make_baseline_cnn
- make_autokeras_image_classifier

### 4. Robust and Notebook-Safe

AutoKeras loading is wrapped in try/except to avoid environment-related crashes.

## 4. How This Notebook Fits Into the Full Project
AutoKeras.API.ipynb (this notebook)

Defines and demonstrates the reusable API layer.

AutoKeras.example.ipynb (Docker notebook)

Runs the full experiment pipeline:
- baseline CNN training
- MobileNetV2 transfer learning
- lightweight AutoKeras search
- evaluation: accuracy, precision, recall, F1
- confusion matrices
- misclassification plots
- saving models and outputs

Colab Notebook (GPU)

Runs the heavy AutoKeras search with:

- higher max_trials
- more epochs
- data augmentation
- GPU acceleration

Best models can be exported back to Docker for evaluation.