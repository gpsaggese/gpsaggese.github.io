# SBERT Example — End-to-End Sentiment Classification

This document walks through a complete example application built using the SBERT internal API.
It shows how cleaned data, SBERT embeddings, and downstream classifiers come together to produce a sentiment model.  

The example covers:  
1.	Dataset loading  
2.	Baseline performance  
3.	SBERT embeddings + classifiers  
4.	Improvements with tuned models and SVM  
5.	Fine-tuning SBERT end-to-end  
6.	Final insights

## 1. Load Data & Embeddings

We begin by loading configuration, cleaned text, and precomputed SBERT embeddings.
```python
cfg = load_config("config.yaml")
df = load_clean_data(cfg)
X = load_embeddings(cfg)
y = df["sentiment_int"].values
```
Dataset summary:  
•	4,846 sentences   
•	3 sentiment classes (negative, neutral, positive)  
•	Embedding dimension: 384  

This verifies preprocessing and embedding stages executed correctly.

## 2. Baseline: Majority Class

Before using any model, we establish a trivial benchmark.  
•	Majority class = 1 (neutral)  
•	Baseline accuracy = ~59%  

Any real model must beat this.

## 3. Logistic Regression on SBERT Embeddings
```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
```
Performance  
•	Accuracy: 0.7629  
•	SBERT embeddings clearly outperform TF-IDF baseline (see next section)  
•	Confusion matrix shows improved separation between classes 0, 1, 2  

## 4. TF-IDF + Logistic Regression (Comparison)

To validate SBERT’s value, we train the same model on TF-IDF.

Results  
•	TF-IDF accuracy: 0.7608  
•	Slightly worse than SBERT  
•	Higher dimensionality (~15k features) but less semantic representation  

Conclusion

SBERT embeddings are compact and slightly more expressive, confirming expected behavior.

## 5. Improved Classifiers on Frozen SBERT

### 5.1 Hyperparameter-tuned Logistic Regression

Grid search over C values:

Best accuracy: 0.7587
(slightly worse than base logistic regression)

### 5.2 Linear SVM (LinearSVC)
```python
svm = LinearSVC(class_weight="balanced")
svm.fit(X_train, y_train)
```
Accuracy: 0.7722
This becomes the best frozen-embedding model.

## 6. Fine-Tuning SBERT (End-to-End)

We train all-MiniLM-L6-v2 directly on the labeled financial text.

Pipeline includes:  
•	Train/validation split  
•	HuggingFace tokenizer  
•	PyTorch Dataset + DataLoader  
•	AdamW optimizer  
•	1 epoch (demo only, already effective)  

Fine-tuning Performance  
•	Accuracy: 0.7907 (best overall)  
•	Stronger recall on minority sentiment classes  
•	Confirms that adapting SBERT to domain-specific text improves performance  

## 8. How to Run This Example
```bash
python src/preprocess.py --config config.yaml
python src/sbert_embed.py --config config.yaml
jupyter notebook SBERT_Example.ipynb
```
## 9. Conclusion

This example demonstrates a complete, reproducible sentiment-analysis workflow using SBERT, including preprocessing, embeddings, classical models, and transformer fine-tuning.
The project’s modular design allows easy experimentation and extension.
