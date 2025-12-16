# 📘 Sentiment Analysis API Documentation

This document describes the **internal programming interface (API)** created for the  
**Sentiment Analysis on Amazon Product Reviews** project.

The purpose of this API is to provide a **clean, reusable, and beginner-friendly abstraction**
for building text-based sentiment classification pipelines using Python.

> ⚠️ This is **not** an external web API.  
> It is an **internal software interface** designed to separate core logic from notebooks and tutorials.

---

## 🎯 Purpose of the API

The Sentiment Analysis API is designed to:

- Encapsulate common NLP operations (cleaning, vectorization, modeling)
- Provide a stable contract for sentiment classification
- Enable reuse across notebooks and applications
- Support both **binary** and **multi-class** sentiment classification

This API is intentionally minimal so that a new user can understand and use it within **60 minutes**.

---

## 🧠 Design Principles

The API follows these principles:

- **Separation of concerns**  
  Core logic lives in Python modules; notebooks remain lightweight.

- **Reusability**  
  Functions and classes can be reused across different datasets.

- **Transparency**  
  Each step of the pipeline is explicit and interpretable.

- **Extensibility**  
  The API can be extended to support deep learning models or new vectorization methods.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    A[Raw Review Text] --> B[Text Cleaning]
    B --> C[TF-IDF Vectorization]
    C --> D[Model Training]
    D --> E[Evaluation]
    D --> F[Prediction]
