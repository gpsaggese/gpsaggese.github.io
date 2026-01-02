# Fashion Product Image Classification – Full Example

This notebook contains the full end-to-end workflow for the Fashion Product Images (Small) dataset. It uses the helper API implemented in `utils_data_io.py` and `utils_model.py` and demonstrates three modeling approaches:

- A baseline CNN trained from scratch  
- A transfer learning model using MobileNetV2 (ImageNet pretrained)  
- A small AutoKeras search (CPU-friendly demo)  

All experiments in this notebook are intended to run **inside Docker**, using local TSV lists and an `images/` folder (as described in `README.md`).

---

## 1. Setup and Dataset Loading

The notebook begins by importing:

- TensorFlow, NumPy, Matplotlib  
- `classification_report` and `confusion_matrix`  
- Helper functions:  
  - `tsv_to_tfds`, `ds_to_numpy` from `utils_data_io.py`  
  - `make_baseline_cnn`, `make_autokeras_image_classifier` from `utils_model.py`

A small TF log level is set to avoid clutter.

Class names used throughout:

- Accessories  
- Apparel  
- Footwear  
- Free Items  
- Personal Care  
- Sporting Goods  

The TSV lists:

- `lists/train.tsv`  
- `lists/val.tsv`  
- `lists/test.tsv`  

These map image file paths to numeric labels.  
`outputs/` and `models/` directories are created if missing.

Load datasets:

Each dataset yields `(224×224×3 image, label)` pairs.

---

## 2. Data Augmentation and tf.data Pipelines

A lightweight augmentation function performs:

- horizontal flips  
- brightness and contrast changes  
- random 90º rotation (0–3)

Using:

This provides efficient GPU-compatible and CPU-friendly pipelines.

---

## 3. Baseline CNN (from scratch)

### 3.1 Prepare NumPy subsets

To limit training time inside Docker:

- MAX_TRAIN = 2000  
- MAX_VAL   = 500  
- MAX_TEST  = 500  

Convert raw datasets to NumPy:

This confirms data loading is correct.

### 3.2 Build baseline model

One function call configures the full CNN:


Architecture (defined in utils_model.py):

- Rescaling  
- Conv2D → MaxPool (x3)  
- Flatten  
- Dense(256) + Dropout  
- Dense(6 softmax)

### 3.3 Train baseline model

Validation accuracy reaches ~93–94%.

### 3.4 Evaluate baseline model

- Predictions made on `X_test`  
- `classification_report` and confusion matrix generated  
- Saved to:


The baseline performs well on major classes but weak on very rare ones.

---

## 4. Transfer Learning with MobileNetV2

### 4.1 Build model

Using ImageNet pretrained MobileNetV2:

- `include_top=False`  
- `weights="imagenet"`  
- Input shape = (224, 224, 3)  
- Frozen backbone  
- Classification head: Rescaling → GAP → Dropout → Dense(6)

### 4.2 Train MobileNetV2


Despite only 3 epochs on CPU:

- Train accuracy ≈ 0.95  
- Validation accuracy ≈ 0.97  

MobileNetV2 significantly outperforms the baseline CNN.

### 4.3 Evaluate MobileNetV2

Using helper `predict_on_dataset`, the test scores are:

- Overall accuracy ≈ 0.97  
- Very strong performance on all major categories

Saved files:


The saved model can be reused without retraining.

---

## 5. AutoKeras Search (Docker demo)

AutoKeras is computationally heavy, so inside Docker I run a **tiny search**:

- AK_MAX_TRAIN = 200  
- AK_MAX_VAL   = 50  
- MAX_TRIALS_AK = 1  
- EPOCHS_AK = 1  

Prepare small NumPy subsets:


Create classifier:


Run minimal search:


### 5.1 Evaluate AutoKeras model


Accuracy ≈ 0.73 (expected due to tiny training subset).

Saved outputs:


A full AutoKeras search will later be run on **Colab Pro GPU**.

---

## 6. Misclassification Visualizations

A helper function shows example failures:


Repeated for:

- Baseline CNN  
- MobileNetV2  
- AutoKeras  

These plots make the differences between models visually clear:

- Baseline: confuses Accessories ↔ Apparel  
- MobileNetV2: far fewer mistakes  
- AutoKeras (tiny search): noticeably weaker  

---

## 7. Running This Notebook in Docker

### Build the image
bash docker_build.sh

### Run JupyterLab
bash docker_jupyter.sh

### Open the notebook
Use the link printed in the terminal, then:


### Open the notebook
Use the link printed in the terminal, then:


### Requirements
`images/` directory must be present locally with the Kaggle dataset extracted.  
TSV files in `lists/` must point correctly to those images.

---

## 8. Relationship to Other Notebooks

### AutoKeras.API.ipynb
Defines the reusable API layer:
- `tsv_to_tfds`, `ds_to_numpy`
- `make_baseline_cnn`, `make_autokeras_image_classifier`

### AutoKeras.example.ipynb (this notebook)
Runs the complete workflow:
- baseline CNN
- MobileNetV2 transfer learning
- AutoKeras CPU demo search
- evaluation + confusion matrices
- misclassification plots

### AutoKeras.example.full_training.ipynb (Colab GPU Notebook)
Will run:
- larger AutoKeras search (more trials + epochs)
- potentially fine-tuned MobileNetV2
- export best models back into Docker environment

## 9. Summary (Docker example vs. Colab full-training notebook)
This notebook keeps the Docker workflow lightweight so it runs reliably on CPU:
- Baseline CNN — trained on a subset (2,000 train / 500 val / 500 test) for 5 epochs.
- MobileNetV2 — trained on the full tf.data pipeline for 3 epochs.
- AutoKeras — limited to max_trials=1 and epochs=1 on a very small NumPy subset, mainly to confirm that the search pipeline and export functionality work.

A larger experiment, including more epochs, bigger subsets, stronger augmentation, and a deeper AutoKeras search, is run separately in Autokeras.example.full_training.ipynb on a GPU environment (Colab).
---

This notebook forms the **main experimental pipeline** of the project and provides all saved outputs.

