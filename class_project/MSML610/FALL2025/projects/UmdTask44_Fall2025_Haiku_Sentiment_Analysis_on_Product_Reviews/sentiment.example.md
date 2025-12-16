# 📗 Sentiment Analysis on Amazon Product Reviews  
### Example Walkthrough (End-to-End Tutorial)

This document accompanies `sentiment.example.ipynb` and demonstrates a **complete, runnable sentiment analysis application** built using the internal Sentiment Analysis API defined in this project.

The goal of this example is to show how the API can be used in practice to load data, preprocess text, train a model, evaluate performance, and make predictions on new product reviews.

---

## 🎯 Objective

Build and evaluate a sentiment classifier for product reviews that categorizes text into:

- **Positive**
- **Negative**
- **Neutral**

The example focuses on clarity and reproducibility rather than maximum model complexity.

---

## 📦 Dataset Overview

We use a subset of the **Amazon Product Reviews** dataset from Kaggle.

Each record typically contains:
- Review text
- Rating score (1–5 stars)
- Product metadata

For this tutorial, we focus only on:
- **Review text**
- **Derived sentiment label**

---

## 🧭 Sentiment Labeling Strategy

To align with the project objective of **three-class sentiment classification**, we define labels as follows:

| Rating | Sentiment |
|------|----------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

This mapping reflects common sentiment conventions and allows the classifier to handle ambiguous or mixed opinions.

> **Note**:  
> While the API fully supports multi-class sentiment classification, the example notebook may also demonstrate a **binary setup (positive vs negative)** for simplicity. The same pipeline extends naturally to three classes.

---

## 🧩 Workflow Overview

The end-to-end workflow demonstrated in this example is:

```mermaid
flowchart TD
    A[Raw Reviews] --> B[Text Cleaning]
    B --> C[TF-IDF Vectorization]
    C --> D[Logistic Regression Training]
    D --> E[Evaluation]
    E --> F[Prediction on New Reviews]


## 🚀 Deployment with Haiku (Real-Time Inference)

To enable real-time sentiment prediction, we wrap the trained
SentimentModel using Haiku.

The model is loaded once and reused for incoming text inputs.
Each request performs:
1. text cleaning
2. TF-IDF transformation
3. sentiment prediction

This design enables low-latency inference suitable for interactive
applications.
