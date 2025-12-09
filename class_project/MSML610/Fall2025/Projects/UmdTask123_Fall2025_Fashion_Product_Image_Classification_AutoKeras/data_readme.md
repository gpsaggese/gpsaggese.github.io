# Data README – Fashion Product Images (Small)

This project uses the **Fashion Product Images (Small)** dataset from Kaggle.  
Because the dataset contains tens of thousands of JPG files, the `images/` folder is **NOT** committed to GitHub.  
This file explains how to correctly set up the dataset locally so the project runs without errors.

---

## 1. Dataset Source (Kaggle)

Dataset used:  
Fashion Product Images (Small)  
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

Download the ZIP file from Kaggle and extract it. The extracted directory will contain all product images (JPG files).

---

## 2. Required Local Folder Structure (Docker version)

After downloading and extracting the dataset, place **ALL images** into this project folder:

UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras/  
│  
├── images/          ← place ALL JPG images here  
├── lists/  
│   ├── train.tsv  
│   ├── val.tsv  
│   └── test.tsv  
│  
└── (rest of the project files)

Important notes:

- The `images/` folder must be **next to** `lists/`, `utils_data_io.py`, and the notebooks.
- Do **not** rename images or move them into subfolders.
- All JPG files must be directly inside the `images/` folder (flat structure).

---

## 3. How TSV Files Work

Each `.tsv` file contains lines in the format:

images/12345.jpg    2  
images/99871.jpg    0  
images/45012.jpg    4  

Column 1: relative path to an image  
Column 2: label index (0–5), where:

0 – Accessories  
1 – Apparel  
2 – Footwear  
3 – Free Items  
4 – Personal Care  
5 – Sporting Goods  

Your local image filenames MUST match exactly those referenced in the TSV files.

---

## 4. Verifying That Data Is Set Up Correctly

After placing the images, inside Docker you can run:

```
from utils_data_io import tsv_to_tfds  
ds = tsv_to_tfds("lists/train.tsv", num_classes=6)
```

If everything is correct, the dataset loads without errors.

If you see:

```
NOT_FOUND: images/12345.jpg; No such file or directory
```

then either:

- the images folder is incomplete,  
- filenames were changed, or  
- the images folder is in the wrong location.

---

## 5. Why images/ Is Not in GitHub

- GitHub does not efficiently store large binary datasets.  
- Kaggle licenses require users to download data themselves.  
- The MSML610 project format expects large datasets to be referenced, not uploaded.

---
## 6. Colab Version – Preparing Data for Full AutoKeras Training
The GPU-intensive notebook: AutoKeras.full_training.ipynb
expects the dataset in Google Drive as: images.zip

Colab notebook will automatically:
- Mount Google Drive
- Unzip images.zip
- Create the same structure used in Docker:

```
/content/images/
lists/train.tsv
lists/val.tsv
lists/test.tsv
```
---
## 7. Quick Checklist for Reviewers (Docker Workflow)

Before running the notebooks:

1. Download dataset from Kaggle  
2. Extract all JPGs into the `images/` folder  
3. Ensure filenames match references in `lists/*.tsv`  
4. Confirm folder structure is correct  
5. Launch Jupyter using Docker:
```
bash docker_jupyter.sh
```
If data is correctly placed, the entire pipeline runs end-to-end inside Docker.

---
## 8. Quick Checklist for Colab (Full GPU Training)

For running AutoKeras.full_training.ipynb:

1. Upload images.zip to Google Drive
2. Open the notebook from Drive or via the provided Colab link
3. Run all cells 
4. Outputs will appear in: 
```
/content/colab_outputs/
```

These files are also included in the project under colab_outputs/.

---

This `data_readme.md` ensures reproducible dataset setup for both Docker and Colab environments.