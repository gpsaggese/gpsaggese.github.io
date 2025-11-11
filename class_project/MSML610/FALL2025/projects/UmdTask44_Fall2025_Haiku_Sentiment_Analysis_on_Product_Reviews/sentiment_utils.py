"""
sentiment_utils.py
Utility functions for Sentiment Analysis project (MSML610).

Keeping reusable logic here: text cleaning, tokenization, vectorization,
model training, evaluation, etc.  All notebooks should import functions
from this file instead of writing raw code inline.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower().strip()

def prepare_data(df, text_col="review", label_col="sentiment"):
    df[text_col] = df[text_col].apply(clean_text)
    return train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

def vectorize(train_texts, test_texts):
    vec = TfidfVectorizer(max_features=5000)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return report
