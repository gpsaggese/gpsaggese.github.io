# AutoKeras API – Fashion Product Image Classification

The purpose of this notebook is not to train the best model.  
Instead, this notebook defines and demonstrates the API layer for the project — a clean, reusable interface for:

- loading the Fashion Product Images dataset, and  
- constructing both the baseline CNN and the AutoKeras ImageClassifier.

This allows the example notebook to focus on full experiments, while this notebook focuses on the design of the API.

---

## 1. API Files Used in This Notebook

### utils_data_io.py — Data Loading Helpers

Functions:

1. tsv_to_tfds(tsv_path, num_classes)  
   - Reads a TSV file (image_path + label_index).  
   - Loads and decodes each image.  
   - Resizes to 224×224 and normalizes pixels.  
   - Returns a tf.data.Dataset of (image_tensor, label).

2. ds_to_numpy(ds, max_samples)  
   - Iterates through a dataset.  
   - Collects up to max_samples.  
   - Returns (X, y) as NumPy arrays.

---

### utils_model.py — Model Builders

Functions:

1. make_baseline_cnn(input_shape, num_classes)  
   - Builds a simple CNN with three Conv+Pool blocks, Flatten, Dense, Dropout, and final softmax layer.  
   - Compiles using Adam and sparse categorical cross-entropy.

2. make_autokeras_image_classifier(num_classes, max_trials)  
   - Returns an AutoKeras ImageClassifier configured with objective="val_accuracy" and a search budget via max_trials.

---

## 2. What This Notebook Demonstrates

### Environment + TensorFlow sanity check

The notebook verifies TensorFlow installation and devices.  
In Docker, the expected output is:

TensorFlow version: 2.20.0  
Available GPUs: []

---

### Loading a small demo batch using the API

To keep execution light, only 64 samples are loaded.  
This validates that TSV parsing, path resolution, and image decoding all work correctly.

Shapes:

X_small → (64, 224, 224, 3)  
y_small → (64,)

A small image grid is plotted for visual confirmation.

---

### Building a baseline CNN using the model API

A single function call constructs the CNN:

baseline_model = make_baseline_cnn(input_shape=X_small.shape[1:], num_classes=NUM_CLASSES)

A short 1-epoch training run confirms that:
- the model compiles correctly, and  
- the data pipeline returns valid tensors.

Full baseline CNN training happens in AutoKeras.example.ipynb.

---

### Creating the AutoKeras classifier

clf = make_autokeras_image_classifier(num_classes=NUM_CLASSES, max_trials=2)

This confirms:
- AutoKeras is installed and functional  
- the wrapper returns a usable classifier  

A full AutoKeras search is not run here because CPU-only Docker is too slow.  
Expected usage is shown as a commented template:

clf.fit(train_ds.batch(32), epochs=5, validation_data=val_ds.batch(32))


---

## 3. Design Decisions

1. Clear separation of concerns  
   - Dataset logic in utils_data_io.py  
   - Model architecture logic in utils_model.py  
   - This notebook only demonstrates usage

2. Docker-friendly execution  
   - Uses small subsets  
   - Keeps runtime low  
   - Heavy training is deferred to example + Colab notebooks

3. Minimal API surface  
   Four functions define the complete API:  
   - tsv_to_tfds  
   - ds_to_numpy  
   - make_baseline_cnn  
   - make_autokeras_image_classifier

4. Robust behavior  
   AutoKeras initialization is wrapped in try/except to avoid crashes.

---

## 4. How This Notebook Fits Into the Full Project

### AutoKeras.API.ipynb (this notebook)
Defines and demonstrates the reusable API components.

### AutoKeras.example.ipynb (Docker)
Runs the full experiment pipeline:
- baseline CNN training  
- MobileNetV2 transfer learning  
- lightweight AutoKeras search  
- evaluation (accuracy, precision, recall, F1)  
- confusion matrices and misclassification plots  
- saving models/results

### Colab notebook (GPU)
Runs the large AutoKeras search with:
- higher max_trials  
- more epochs  
- data augmentation  
- GPU acceleration  

Best models can be exported back to Docker for evaluation.

---

This notebook establishes the core API layer for the Fashion Product Image Classification project, ensuring all other notebooks use a consistent and maintainable workflow.
