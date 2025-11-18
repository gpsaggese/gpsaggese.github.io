---
# 📰 Fake News Detection using Fine-Tuned RoBERTa

## 📘 Overview

This project develops a **Fake News Detection** system using **Natural Language Processing (NLP)** and a **fine-tuned RoBERTa transformer model**.
The model classifies news articles as **True** or **Fake** based on textual content.
It was completed as part of the **MSML610 — Applied Machine Learning** course at the **University of Maryland, College Park**.

The project integrates **data preprocessing**, **Transformer fine-tuning**, **evaluation**, and **explainability** using **SHAP** and **LIME**, ensuring both high predictive accuracy and interpretability.
---

## 🧠 Objectives

- Fine-tune **RoBERTa-base** for domain-specific fake news classification.
- Apply **text preprocessing**, **tokenization**, and **class balancing (SMOTE)**.
- Evaluate model performance across multiple metrics.
- Use **SHAP** and **LIME** to interpret predictions at the token level.
- Visualize model performance and word importance to enhance transparency.

---

## 📂 Dataset

**Source:** [Kaggle – Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)

**Files Used:**

- `fake.csv` — Fake news articles
- `true.csv` — Genuine news articles

**Google Drive Path:**

```
/content/drive/MyDrive/MSML610/News_dataset/
```

**Features:**

- `title` — News headline
- `text` — Full article content
- `subject` — Category of article
- `date` — Publication date
- `label` — 0 = Fake, 1 = True

---

## ⚙️ Data Preparation

1. Loaded both datasets and assigned binary labels.
2. Combined and shuffled data to ensure random distribution.
3. Cleaned text (removed HTML tags, punctuation, URLs, and stopwords).
4. Split data into **train (80%)**, **validation (10%)**, and **test (10%)**.
5. Addressed imbalance using **SMOTE** to synthesize minority samples.

---

## 🧩 Model Architecture

- **Base Model:** `roberta-base` (from Hugging Face Transformers)
- **Fine-Tuning Framework:** PyTorch + Hugging Face `Trainer` API
- **Tokenizer:** `RobertaTokenizer.from_pretrained('roberta-base')`
- **Classifier Head:** Linear layer + Softmax activation
- **Optimizer:** AdamW
- **Loss Function:** CrossEntropyLoss
- **Training Configuration:**

  - Batch size: 16
  - Epochs: 3–5
  - Learning rate: 2e-5
  - Evaluation strategy: `epoch`

---

## 🔧 Implementation Environment

**Platform:** Google Colab (with GPU)
**Python Version:** 3.10+

**Install Dependencies**

```python
%pip install -U "transformers" "datasets" "peft" "accelerate" "torch" "scikit-learn" "lime" "shap" --quiet
```

**Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧪 Fine-Tuning Workflow

1. **Tokenization:** Encode text with `RobertaTokenizer`.
2. **Dataset Conversion:** Convert Pandas DataFrame → Hugging Face `DatasetDict`.
3. **Model Loading:** `RobertaForSequenceClassification(num_labels=2)`.
4. **Training:** Fine-tune model using `Trainer` with evaluation after each epoch.
5. **Model Saving:** Export fine-tuned weights to Google Drive for reuse.

**Example Code Snippet**

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

---

## 📈 Model Performance

**Overall Performance Metrics:**

| Metric        | Score      |
| ------------- | ---------- |
| **Accuracy**  | **0.9824** |
| **Precision** | **0.9903** |
| **Recall**    | **0.9759** |
| **F1-score**  | **0.9831** |
| **AUC-ROC**   | **0.9980** |

**Classification Report:**

| Label            | Precision | Recall | F1-Score   | Support  |
| ---------------- | --------- | ------ | ---------- | -------- |
| 0 (Fake)         | 0.9740    | 0.9895 | 0.9817     | 4284     |
| 1 (True)         | 0.9903    | 0.9759 | 0.9831     | 4696     |
| **Accuracy**     |           |        | **0.9824** | **8980** |
| **Macro avg**    | 0.9822    | 0.9827 | 0.9824     | 8980     |
| **Weighted avg** | 0.9825    | 0.9824 | 0.9824     | 8980     |

**Interpretation:**
The fine-tuned RoBERTa model achieved **98.2% accuracy**, with a near-perfect **AUC-ROC of 0.998**, indicating extremely strong separability between fake and true news classes. Both precision and recall remain well-balanced, confirming robustness against overfitting or bias toward a particular class.

---

## 🔍 Explainability

To ensure model transparency and interpretability, **SHAP** and **LIME** were used post-training.

### 🧩 SHAP (SHapley Additive exPlanations)

- Quantifies each token’s contribution to the final prediction.
- Generates global and local word importance plots.
- Confirms that high-impact terms (e.g., emotionally charged or politically biased words) drive “fake” classifications.

### 💡 LIME (Local Interpretable Model-agnostic Explanations)

- Perturbs text inputs and observes model sensitivity.
- Visualizes token influence for individual predictions.
- Useful for debugging and validating local decision consistency.

These tools reveal that the model’s predictions align with logical linguistic cues rather than random noise — a key aspect of **ethical and explainable AI**.

---

## 📊 Visualization Outputs

The notebook includes:

- Training vs. Validation Loss Curves
- Confusion Matrix Heatmap
- SHAP Token Importance Visualizations
- LIME Local Explanation Charts

---

## 🧾 References

- Liu, Y. et al. (2019). _RoBERTa: A Robustly Optimized BERT Pretraining Approach._
- Ribeiro, M. T. et al. (2016). _“Why Should I Trust You?”: Explaining the Predictions of Any Classifier._
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)

---
