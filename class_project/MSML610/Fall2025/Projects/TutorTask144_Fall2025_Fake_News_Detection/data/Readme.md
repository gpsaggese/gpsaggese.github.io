# Fake News Detection Dataset

This repository uses publicly available datasets for **fake news detection**, along with a preprocessed version generated through a dedicated preprocessing pipeline.

---

## Datasets Overview

The project is based on **two primary datasets**:

- **`fake.csv`** — News articles labeled as *fake*
- **`true.csv`** — News articles labeled as *real*

---

##  Dataset Download Options

You may obtain the datasets using **either** of the following methods.

### Option 1: Kaggle (Recommended)

Download both datasets from Kaggle:

- https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

Included files:
- `fake.csv`
- `true.csv`

---

### Option 2: Google Drive (Direct Download)

- **true.csv**  
  https://drive.google.com/file/d/1QlbN0U5DcK-m0EUfAoyZjpDZu4qzAoPf/view?usp=sharing

- **fake.csv**  
  https://drive.google.com/file/d/1tHZhc6xYk_Q9oDPClMnUawJD3V_4tOBr/view?usp=sharing

---

## Preprocessed Dataset

A cleaned and merged dataset is provided for convenience.

- **preprocessed_data.csv**  
  https://drive.google.com/file/d/1DZG7hdSuWW6U6wXIMy5UUK1hS8H-WYxt/view?usp=sharing

### How it was created

The file was generated using the notebook **Data_Preprocessing.ipynb**, which performs:

- Label standardization (fake vs. real)
- Text cleaning and normalization
- Removal of duplicates and missing values
- Dataset merging

You can regenerate this file by running `Data_Preprocessing.ipynb`.

---

## Directory Structure

```text
project-root/
│
├── data/
│   ├── fake.csv
│   ├── true.csv
│   └── preprocessed_data.csv
│
├── notebooks/
│   └── Data_Preprocessing.ipynb
│
└── README.md
