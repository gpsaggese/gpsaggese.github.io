
# SBERT Financial Sentiment Classification — Experimental Summary

This document provides a narrative explanation of the experiments performed in
`SBERT_Example.ipynb`. It compares SBERT sentence embeddings with a traditional
TF-IDF baseline and evaluates multiple classification models.

The dataset used is the **Financial PhraseBank**, a standard benchmark for
three-way sentiment classification:
- **0** → Negative  
- **1** → Neutral  
- **2** → Positive  

---

## 1. Data Preparation

### Cleaning
Using `preprocess.py`, the raw CSV is cleaned by:
- keeping only text and label columns,
- removing empty sentences,
- standardizing whitespace,
- mapping labels to integer classes.

Final dataset summary:
- **Total samples:** 4,846  
- **Label distribution:** moderately imbalanced, dominated by class 1 (neutral)

### Embedding Generation
SBERT (`all-MiniLM-L6-v2`) produces a **384-dim embedding** per sentence:
embedding shape = (4846, 384)

These embeddings capture semantic meaning rather than keyword overlap.

---

## 2. Baseline: TF-IDF + Logistic Regression

A TF-IDF model was built using:
- unigrams + bigrams  
- vocabulary capped at 20,000 terms  
- minimum document frequency = 2  

Model performance:

| Metric | Score |
|--------|--------|
| Test Accuracy | **~74%** |
| Macro F1 | ~0.71 |

TF-IDF performs reasonably well due to the dataset’s short financial sentences,
but struggles with semantic nuances and paraphrases.

---

## 3. SBERT + Logistic Regression

SBERT embeddings were used with the same classifier.

Performance:

| Metric | Score |
|--------|--------|
| Test Accuracy | **~76%** |
| Macro F1 | ~0.72–0.73 |

Key findings:
- SBERT improves accuracy by **~2%** over TF-IDF.
- Neutral sentences remain easiest; positive/negative distinction is harder.
- Confusion mainly occurs between positive ↔ neutral.

SBERT provides cleaner separation because its embeddings encode contextual
meaning, not just surface-level tokens.

---

## 4. SBERT + Linear SVM

A linear SVM performs slightly better on high-dimensional dense embeddings.

Performance:  
- Accuracy: **76–77%**  
- Most gains appear in the negative class due to better margin separation.

This model is strong and computationally cheap.

---

## 5. Fine-Tuning a Transformer Classifier (Bonus)

A lightweight transformer (`distilbert-base-uncased`) was fine-tuned for
3-class sentiment.

Training details:
- 2 epochs  
- batch size 16  
- learning rate = 2e-5  
- AdamW optimizer  

Results:
- Validation accuracy: **~78%**
- Slight gains across all classes, especially positive sentiment.

Fine-tuning outperforms frozen SBERT because the model adapts to financial
domain-specific patterns.

---

## 6. Summary of Findings

| Method | Accuracy | Notes |
|--------|----------|-------|
| **TF-IDF + LR** | ~74% | strong lexical baseline |
| **SBERT + LR** | ~76% | improvement via semantic embeddings |
| **SBERT + Linear SVM** | ~76–77% | better margin-based separation |
| **Fine-tuned transformer** | ~78% | best performance, domain-adapted |

---

## 7. Recommendations & Future Work

- **Try domain-specific SBERT models** (FinBERT or Financial-SBERT).
- **Perform hyperparameter search** for SVM and logistic regression.
- **Augment training with unlabeled financial text** (continued pre-training).
- **Explore contrastive learning** to strengthen embedding quality.

---

This document serves as the narrative companion to the example notebook and
summarizes the experimental outcomes clearly and reproducibly.