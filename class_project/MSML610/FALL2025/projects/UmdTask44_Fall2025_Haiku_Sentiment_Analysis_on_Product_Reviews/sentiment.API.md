
---

##  3. sentiment.API.md
```markdown
#  Sentiment Analysis API Documentation

This document describes the internal programming interface (API) created for the **Sentiment Analysis on Product Reviews** tutorial.

---

##  Purpose
The API provides a clean, beginner-friendly abstraction around text preprocessing, vectorization, model training, and prediction.

---

##  Architecture Overview
```mermaid
flowchart TD
A[Raw Review Data] --> B[Preprocessing - clean_text()]
B --> C[Vectorization - TF-IDF]
C --> D[Model Training - Logistic Regression]
D --> E[Evaluation & Prediction]
