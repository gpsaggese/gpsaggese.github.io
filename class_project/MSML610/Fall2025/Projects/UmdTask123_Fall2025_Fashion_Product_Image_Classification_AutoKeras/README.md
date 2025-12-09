# Fashion Product Image Classification – MSML610 Class Project

This project implements an end-to-end image classification pipeline for the **Fashion Product Images (Small)** dataset. It compares three modeling strategies:

- **Baseline CNN** (trained from scratch)  
- **Transfer Learning** with MobileNetV2  
- **AutoKeras Neural Architecture Search**  
  - *Small search inside Docker*  
  - *Full GPU search in Colab for final results*

All deliverables follow MSML610 project requirements.  
The TA can run every experiment using **Docker only** (no GPU needed), can also run Colab version but it takes time.

---

# 1. Project Structure

```
UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras/
│
├── AutoKeras.API.ipynb
├── AutoKeras.API.md
│
├── AutoKeras.example.ipynb
├── AutoKeras.example.md
│
├── AutoKeras.full_training.ipynb        ← full AutoKeras GPU search (Colab)
├── colab_outputs/                       ← all outputs from Colab GPU run
│   ├── baseline_cnn_full_colab.keras
│   ├── mobilenetv2_fashion_colab.keras
│   ├── autokeras_best_colab.keras
│   ├── report_baseline_cnn.txt
│   ├── report_mobilenetv2.txt
│   ├── report_autokeras_colab.txt
│   ├── confmat_baseline_cnn.txt
│   ├── confmat_mobilenetv2.txt
│   ├── confmat_autokeras_colab.txt
│   └── (additional saved files)
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
├── images/                     ← NOT pushed to GitHub
│
├── outputs/                    ← outputs from Docker example notebook
│   ├── report_baseline.txt
│   ├── report_mobilenet.txt
│   ├── report_autokeras.txt
│   ├── confmat_baseline.png
│   ├── confmat_mobilenet.png
│   ├── confmat_autokeras.png
│
└── models/                     ← models saved from Docker example
    ├── mobilenetv2_fashion.keras
    ├── autokeras_best_docker.keras
    └── (other optional models)
```

---

# 2. Notebook Descriptions

## **AutoKeras.API.ipynb**
Defines the project’s reusable API:
- `tsv_to_tfds()`
- `ds_to_numpy()`
- `make_baseline_cnn()`
- `make_autokeras_image_classifier()`

Lightweight notebook used to demonstrate data flow + model creation.

---

## **AutoKeras.example.ipynb** (main Docker notebook)
This is the **full experiment pipeline**, fully reproducible by the TA:

- dataset loading via TSV  
- augmentation  
- baseline CNN training  
- MobileNetV2 transfer learning  
- **small AutoKeras search (CPU-friendly)**  
- evaluation metrics  
- confusion matrices  
- misclassified image visualization  
- saving outputs + models  

**TA should run this notebook.**

---

## **AutoKeras.full_training.ipynb** (Colab GPU)
Runs the **large-scale AutoKeras search**:

- higher `max_trials`  
- more epochs  
- larger subsets  
- GPU acceleration  
- exports best models + reports  

Outputs are stored in:

```
colab_outputs/
```

---

# 3. How the TA Should Run the Project (Docker Only)

### **Step 1 — Build Docker image**
```bash
bash docker_build.sh
```

### **Step 2 — Start an interactive container**
```bash
bash docker_bash.sh
```

### **Step 3 — Launch JupyterLab**
```bash
bash docker_jupyter.sh
```

Open the printed URL in the browser.

### **Step 4 — Run notebooks**
Inside Jupyter:

1. Run **AutoKeras.API.ipynb**  
   (sanity check — verifies dataset loading + model factory functions)

2. Run **AutoKeras.example.ipynb**  
   (main pipeline — Baseline CNN, MobileNetV2, AutoKeras CPU demo, evaluation, outputs)

---

### **Optional: Full AutoKeras Training on Colab Pro**

- **AutoKeras.full_training.ipynb**  
  (runs the full GPU search: higher max_trials, more epochs, heavier models)

A **direct Colab link** is included inside this project file so anyone can open and run it instantly on **Colab Pro / Pro+** if they want to reproduce the full AutoKeras GPU results.

---

# 4. Dataset Requirements

Before running Docker, you must download the **Fashion Product Images (Small)** dataset from Kaggle.

### Local (Docker)
- The `images/` folder **is NOT included in GitHub** because the dataset is large.
- You must manually download and extract the dataset locally so the structure becomes:
```
  images/
      000001.jpg
      000002.jpg
      ...

- The TSV files must point to these paths:
  - lists/train.tsv
  - lists/val.tsv
  - lists/test.tsv

Docker notebooks will read images directly from this local folder.

---

### Google Colab (Full AutoKeras Training)

For running **AutoKeras.full_training.ipynb**:

- Upload the dataset to Google Drive as **images.zip**.
- The Colab notebook automatically mounts Drive and unzips the file.
- This keeps the notebook clean and avoids storing the dataset in GitHub.

This way:
- Docker has direct access to the unzipped dataset.
- Colab loads the zipped version from Drive for heavy GPU training.

---

# 5. Outputs & Models

### Docker Example Notebook Produces:
```
outputs/report_baseline.txt
outputs/report_mobilenet.txt
outputs/report_autokeras.txt
outputs/confmat_baseline.png
outputs/confmat_mobilenet.png
outputs/confmat_autokeras.png

models/mobilenetv2_fashion.keras
models/autokeras_best_docker.keras
```

### Colab GPU Notebook Produces (Final Results):
```
colab_outputs/baseline_cnn_full_colab.keras
colab_outputs/mobilenetv2_fashion_colab.keras
colab_outputs/autokeras_best_colab.keras
colab_outputs/report_baseline_cnn.txt
colab_outputs/report_mobilenetv2.txt
colab_outputs/report_autokeras_colab.txt
colab_outputs/confmat_baseline_cnn.txt
colab_outputs/confmat_mobilenetv2.txt
colab_outputs/confmat_autokeras_colab.txt
```

These represent the **highest accuracy models**.

---
# 6. Gradio Demo (Colab Only)

This project includes a small **Gradio demo app** that allows uploading a product image and viewing the predicted category using the trained MobileNetV2 model.

- The Gradio code is included inside **AutoKeras.full_training.ipynb**.
- The demo is meant to run **only on Google Colab**.

A direct link to the Colab notebook is provided in the "AutoKeras.full_training.ipynb" so you can open the file and view the Gradio output exactly as it appeared during training.  
This Gradio app is included only as a showcase of how the trained model can be deployed interactively.

# 7. Relationship Between Notebooks

| Notebook | Purpose |
|---------|---------|
| AutoKeras.API.ipynb | API demonstration |
| AutoKeras.example.ipynb | Full CPU pipeline | 
| AutoKeras.full_training.ipynb | GPU AutoKeras search | 

---

# 8. Author
**Lokesh Reddy Konda**  
MSML610 – Fall 2025  
University of Maryland

---

This repository includes all deliverables required for the MSML610 class project:  
Docker setup, notebooks, API code, GPU-trained models, outputs, and documentation.
