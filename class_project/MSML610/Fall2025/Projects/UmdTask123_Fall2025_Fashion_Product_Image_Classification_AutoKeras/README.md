# Fashion Product Image Classification – MSML610 Class Project

This project implements an end-to-end image classification pipeline using the **Fashion Product Images (Small)** dataset.  
It compares three different modeling strategies:

- **Baseline CNN** (trained from scratch)  
- **Transfer Learning** with MobileNetV2 (ImageNet pretrained)  
- **AutoKeras Search** (small version inside Docker, full version on Colab GPU)

All experiments run inside a **Docker container**, following the MSML610 required project structure.

---

## Project Structure

UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras/  
│  
├── AutoKeras.API.ipynb  
├── AutoKeras.API.md  
│  
├── AutoKeras.example.ipynb  
├── AutoKeras.example.md  
│  
├── utils_data_io.py  
├── utils_model.py  
│  
├── Dockerfile  
├── docker_build.sh  
├── docker_bash.sh  
├── docker_jupyter.sh  
│  
├── README.md  
├── data_readme.md  
│  
├── lists/  
│   ├── train.tsv  
│   ├── val.tsv  
│   └── test.tsv  
│  
├── images/                  (local only; NOT committed to GitHub)  
│  
├── outputs/                 (classification reports, confusion matrices)  
└── models/                  (saved Keras + AutoKeras models)

---

## API Utilities

### `utils_data_io.py`
- `tsv_to_tfds()` — loads images + labels from TSV lists into a `tf.data.Dataset`  
- `ds_to_numpy()` — extracts up to N samples into NumPy arrays  

### `utils_model.py`
- `make_baseline_cnn()` — factory for the baseline CNN  
- `make_autokeras_image_classifier()` — wrapper to instantiate AutoKeras ImageClassifier  

### `AutoKeras.API.ipynb` / `.md`
Demonstrates how the API layer works (no heavy training).  
Useful for understanding dataset flow and model construction.

---

## Full Workflow Notebook

### `AutoKeras.example.ipynb` / `.md`
This is the main experimental notebook.  
It performs the complete pipeline:

- Load dataset via TSV  
- Apply augmentation  
- Train Baseline CNN  
- Train MobileNetV2 transfer learning model  
- Run a lightweight AutoKeras CPU search  
- Evaluate: precision, recall, F1, accuracy  
- Generate confusion matrices  
- Visualize misclassified samples  
- Save outputs to `outputs/` and `models/`

---

## How to Reproduce Results (Recommended Workflow)

1. **Build Docker image**
```bash
bash docker_build.sh
```

2. **Start interactive container**
```bash
bash docker_bash.sh
```

3. **Launch JupyterLab**
```bash
bash docker_jupyter.sh
```

4. Open the URL printed in the terminal to access JupyterLab.

5. In the browser:
- Run **AutoKeras.API.ipynb** (checks API + debugging)
- Run **AutoKeras.example.ipynb** (full experiments)

---

## Requirements

- An `images/` directory **must exist locally** with the Kaggle dataset extracted  
- `lists/train.tsv`, `lists/val.tsv`, `lists/test.tsv` must contain correct relative paths  
- `outputs/` and `models/` are created automatically  

(Important: **images/ is NOT tracked in GitHub**.)

---

## AutoKeras Full Search (Colab GPU)

Since AutoKeras is slow on CPU-only Docker, a **full hyperparameter search** is run on **Colab Pro GPU**, where we can increase:

- `max_trials`  
- training epochs  
- model complexity  

Exported Colab models (e.g., `.keras`, `.h5`) can be placed directly into:
models/


Docker notebooks will load them if present.

---

## Outputs Generated

Running `AutoKeras.example.ipynb` produces:

- `outputs/report_baseline.txt`  
- `outputs/report_mobilenet.txt`  
- `outputs/report_autokeras.txt`  
- `outputs/confmat_baseline.png`  
- `outputs/confmat_mobilenet.png`  
- `outputs/confmat_autokeras.png`  
- `models/mobilenetv2_fashion.keras`  
- `models/autokeras_best_docker.keras`

All these files are used in documentation and the final project video.

---

## Relationship to Other Notebooks

### AutoKeras.API.ipynb  
Defines the reusable components (dataset loader + model builders).

### AutoKeras.example.ipynb  
Full experimental pipeline and evaluation.

### Colab Notebook  
Heavy AutoKeras search + optional fine-tuning of MobileNetV2.

---

## Author

**Lokesh Reddy Konda**  
MSML610 – Fall 2025  
University of Maryland  

---

This repository contains all deliverables required for the MSML610 class project:  
Docker setup, code, documentation, notebooks, saved models, and experiment outputs.


