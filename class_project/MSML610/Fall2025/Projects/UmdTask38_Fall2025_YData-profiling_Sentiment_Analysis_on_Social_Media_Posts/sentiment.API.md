# Sentiment Analysis API – Twitter Airline Tweets

## 1. What this API does

This project builds a small sentiment analysis API for the **Twitter US Airline
Sentiment** dataset. The idea is:

- `sentiment.example.ipynb` trains a model (TF–IDF + Logistic Regression),
  evaluates it, and saves the artifacts.
- `sentiment.API.ipynb` loads the saved model and behaves like a client or
  "API user": it takes raw text and returns sentiment labels.

The main Python code lives in `sentiment_utils.py`. That file is meant to be
reusable in other notebooks or services.

---

## 2. Files

Inside this project folder:

- `data/Tweets.csv` – raw dataset of airline tweets.
- `sentiment_utils.py` – helper functions for loading, cleaning, splitting,
  training, evaluation, and prediction.
- `sentiment.example.ipynb` – full training + evaluation notebook.
- `sentiment.API.ipynb` – notebook that loads the trained model and exposes a
  simple prediction interface.
- `tfidf_vectorizer.joblib` – saved TF–IDF vectorizer.
- `logreg_sentiment_model.joblib` – saved Logistic Regression model.

---

## 3. High-level pipeline

The logical pipeline implemented by `sentiment_utils.py` and the notebooks is:

```text
Tweets.csv
   ↓
load_data
   ↓
preprocess_dataframe
   ↓
split_data
   ↓
vectorize_and_train
   ↓
evaluate_model
   ↓
predict_sentiment / predict_sentiment_api
