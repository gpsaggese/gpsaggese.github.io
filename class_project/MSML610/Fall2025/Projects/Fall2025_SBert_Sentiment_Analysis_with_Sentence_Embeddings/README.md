# SBERT Sentiment Analysis on Financial PhraseBank (MSML610 – Fall 2025)

This project implements an end-to-end sentiment classification pipeline using
**Sentence-BERT (SBERT)** on the **Financial PhraseBank** dataset.
It includes data preprocessing, embedding generation, baseline lexical models,
embedding-based classifiers, and transformer fine-tuning.

The project is designed to be **reproducible, modular, and well-documented**,
with both script-based pipelines and notebook-based demonstrations.

---

## 1. Project Structure
Fall2025_SBert_Sentiment_Analysis_with_Sentence_Embeddings/  
│  
├── README.md  
├── LICENSE_DATASET.md  
├── config.yaml  
│  
├── data/  
│   ├── raw/            # raw dataset (not included)  
│   └── processed/      # cleaned CSV, labels.npy, embeddings.npy  
│  
├── src/  
│   ├── preprocess.py  
│   ├── sbert_embed.py  
│   └── SBERT_utils.py  
│  
└── docs/    
├── SBERT_API.md   
├── SBERT_API.ipynb  
├── SBERT_Example.ipynb  
├── SBERT_Example.md  
├── requirements.txt   
├── Dockerfile    

---

## 2. Environment Setup

### **Option A — Local Environment (recommended)**

Activate the course helper environment:

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

## 4. Documentation & Notebooks

### SBERT_API.ipynb / SBERT_API.md

Documents the internal API layer used throughout the project:   
•	configuration handling  
•	data loading utilities  
•	embedding generation  
•	reusable SBERT inference helpers  

### SBERT_Example.ipynb / SBERT_Example.md

Demonstrates the end-to-end sentiment analysis workflow, including:  
•	TF-IDF + Logistic Regression baseline  
•	Frozen SBERT + Logistic Regression  
•	Frozen SBERT + Linear SVM  
•	End-to-end fine-tuned transformer classifier  
•	Accuracy, F1-score, and confusion matrices  

These files represent the primary submission artifacts for API usage
and example-driven documentation.

## 5. Docker (Reproducibility)

To build the container:
```bash
docker build -t sbert-sentiment .
```

Run Jupyter Lab inside Docker:
```bash
docker run -p 8888:8888 sbert-sentiment
```
Open a browser and navigate to the Jupyter server exposed on port 8888:

http://localhost:8888

Depending on the Jupyter configuration, this may open:  
- the JupyterLab interface, or  
- the classic notebook file browser.  

In either case, the project notebooks can be accessed directly from the browser.

This ensures full reproducibility independent of local Python installations.

## 6. Results Summary

TF-IDF + Logistic Regression - Accuracy ~0.74

SBERT + Logistic Regression - Accuracy ~0.76

SBERT + Linear SVM - Accuracy ~0.76–0.77

Fine-tuned Transformer - Accuracy ~0.78

Embedding-based models consistently outperform lexical baselines,
with the fine-tuned transformer achieving the strongest overall performance.
