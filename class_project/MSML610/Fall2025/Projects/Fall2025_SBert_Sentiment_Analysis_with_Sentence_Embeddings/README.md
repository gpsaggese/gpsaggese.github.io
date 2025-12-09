# SBERT Sentiment Analysis on Financial PhraseBank

This project implements a full sentiment-classification pipeline using **Sentence-BERT (SBERT)** embeddings on the *Financial PhraseBank* dataset.  
It includes data preprocessing, embedding generation, baseline models, SBERT-based classifiers, and optional transformer fine-tuning.

This project is submitted as part of **MSML610 — Fall 2025**.

---

## 1. Project Structure
Fall2025_SBert_Sentiment_Analysis_with_Sentence_Embeddings/
│
├── README.md
├── LICENSE_DATASET.md
├── config.yaml
│
├── data/
│   ├── raw/          # raw dataset (not included)
│   └── processed/    # cleaned CSV, labels.npy, embeddings.npy
│
├── src/
│   ├── preprocess.py
│   ├── sbert_embed.py
│   └── SBERT_utils.py
│
└── docs/
├── SBERT_API.md
└── SBERT_Example.md

---

## 2. Setup Instructions

### **Option A — Local Environment (recommended)**

Create and activate the course virtual environment:

```bash
source client_venv.helpers/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## 3. Running the Pipeline

### Step 1 — Preprocess the dataset
```bash
python src/preprocess.py --config config.yaml
```
Outputs:
	•	data/processed/financial_phrasebank_clean.csv
	•	data/processed/labels.npy

### Step 2 — Generate SBERT embeddings
```bash
python src/sbert_embed.py --config config.yaml
```
Outputs:
	•	data/processed/sbert_embeddings.npy
(shape: N × 384, where N = number of sentences)

## 4. Notebooks

Two documentation notebooks are included:

✔ SBERT_API.ipynb / SBERT_API.md

Explains:
	•	configuration usage
	•	loading cleaned data
	•	loading embeddings
	•	how inference works with SBERT

✔ SBERT_Example.ipynb / SBERT_Example.md

Covers:
	•	baseline TF-IDF model
	•	SBERT + Logistic Regression
	•	SBERT + Linear SVM
	•	optional fine-tuned transformer classifier
	•	metrics, confusion matrix, cross-validation results

This notebook demonstrates the full comparison between lexical vs. embedding-based models.

## 5. Docker Support

To build the container:
```bash
docker build -t sbert-sentiment .
```

Run Jupyter Lab inside Docker:
```bash
docker run -p 8888:8888 sbert-sentiment
```
Then open:
http://127.0.0.1:8888/lab

This ensures full reproducibility independent of local Python installations.

## 6. Results Summary

TF-IDF + Logistic Regression - Accuracy ~0.74

SBERT + Logistic Regression - Accuracy ~0.76

SBERT + Linear SVM - Accuracy ~0.76–0.77

Fine-tuned Transformer - Accuracy ~0.78

