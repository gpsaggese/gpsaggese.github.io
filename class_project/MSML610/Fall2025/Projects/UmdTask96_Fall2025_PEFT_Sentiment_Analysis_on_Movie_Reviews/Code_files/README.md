# 📘 **README.md**

# **PEFT Sentiment Analysis on Movie Reviews**

### _(Fake vs True News Classification Using RoBERTa + LoRA)_

This project is part of the **“Learn X in 60 Minutes”** tutorial series.
The goal is to introduce a beginner-friendly, reproducible, hands-on pipeline for fine-tuning transformer models using **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** adapters.

Although the project name references _movie reviews_, this tutorial demonstrates binary text classification using the **Fake vs True News dataset**, which follows the same structure as sentiment analysis tasks.

All logic is fully reproducible using Docker and runs cleanly via **Restart & Run All** in the included notebooks.

---

# 🚀 **Project Objectives**

This tutorial teaches the following in under 60 minutes:

- How to load and preprocess raw text data
- How to convert data into HuggingFace Datasets
- How to tokenize using RoBERTa
- How to apply **LoRA** adapters to reduce fine-tuning cost
- How to train using the HuggingFace Trainer API
- How to evaluate with precision, recall, F1, ROC-AUC
- How to use SHAP for explainability
- How to structure your project using an API + utils + examples approach

This tutorial is designed for beginners and intermediate practitioners who want a fast, well-structured introduction to PEFT.

---

# 📂 **Project Structure**

```
PEFT_Sentiment_Analysis_on_Movie_Reviews/
│
├── PEFT_Sentiment_Analysis_on_Movie_Reviews_utils.py
│   → All reusable logic, data loaders, preprocessing, tokenization,
│     model loading, LoRA config, training setup, evaluation, SHAP.
│
├── PEFT_Sentiment_Analysis_on_Movie_Reviews.API.md
│   → Documentation of native APIs + wrapper APIs + architecture.
│
├── PEFT_Sentiment_Analysis_on_Movie_Reviews.API.ipynb
│   → Minimal notebook demonstrating the API surface.
│
├── PEFT_Sentiment_Analysis_on_Movie_Reviews.example.md
│   → Full example walkthrough with diagrams (no code).
│
├── PEFT_Sentiment_Analysis_on_Movie_Reviews.example.ipynb
│   → Complete end-to-end training & evaluation pipeline.
│
├── Dockerfile
│   → Reproducible environment, CPU-friendly, installs all dependencies.
│
└── README.md
    → You are reading this file.
```

---

# 📊 **Technologies Used**

### **Libraries**

- HuggingFace Transformers
- HuggingFace Datasets
- PEFT (LoRA adapters)
- PyTorch
- NLTK (tokenization, lemmatization)
- Scikit-Learn (metrics)
- Imbalanced-Learn (SMOTE)
- SHAP & LIME (explainability)
- Seaborn & Matplotlib (visualization)

### **Models**

- `roberta-base`
- LoRA-adapted RoBERTa for binary classification

---

# 🐳 **Docker Setup**

This project is fully containerized.
No local environment setup needed.

### **1. Build the Docker image**

```
docker build -t peft_sentiment .
```

### **2. Run the container**

```
docker run -p 8888:8888 -v $(pwd):/app peft_sentiment
```

### **3. Open the notebook**

Navigate to:

```
http://127.0.0.1:8888
```

Jupyter will launch automatically with **no token/password required**.

---

# 📘 **How to Use This Tutorial**

### **Step 1 — Read the API documentation**

Start with:

```
PEFT_Sentiment_Analysis_on_Movie_Reviews.API.md
```

This explains:

- native APIs (HF + PEFT)
- wrapper APIs (utils functions)
- design decisions
- architecture diagrams

---

### **Step 2 — Explore the API notebook**

```
PEFT_Sentiment_Analysis_on_Movie_Reviews.API.ipynb
```

This notebook:

- demonstrates tokenizer
- demonstrates dataset API
- demonstrates LoRA config
- shows how the utils API is used
- DOES NOT train a model

This is your “learn the tool” portion.

---

### **Step 3 — Run the full example pipeline**

```
PEFT_Sentiment_Analysis_on_Movie_Reviews.example.ipynb
```

This notebook:

- loads data
- preprocesses text
- tokenizes for RoBERTa
- applies LoRA
- trains for 800 steps
- evaluates
- plots confusion matrix
- runs SHAP explainability

This is the **complete end-to-end project**.

---

# 📈 **Expected Outputs**

- Clean, reproducible training performance
- Accuracy, precision, recall, F1-score, ROC-AUC
- Confusion matrix visualization
- SHAP token-level explanation
- Demo of PEFT reducing trainable parameters dramatically (~0.7%)

---

# 💡 **Why LoRA?**

Traditional fine-tuning:

- trains 100+ million parameters
- needs GPUs
- expensive & slow

LoRA:

- trains < 1% of parameters
- works on CPU
- extremely fast
- maintains accuracy

This project demonstrates how to adopt **efficient NLP training techniques** for real-world applications.

---

# 📚 **References**

- HuggingFace Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- HuggingFace Datasets: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
- PEFT & LoRA: [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- RoBERTa Paper: Liu et al., 2019
- SHAP Documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)

---
