"""
Sentiment Analysis Utilities

Core functions for the sentiment analysis pipeline on airline tweets.
Handles data loading, preprocessing, vectorization, model training, and prediction.
"""

import re
from typing import Tuple, Dict, List, Union
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Label mapping - keep it simple and consistent
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
INT_FROM_STR = {v: k for k, v in LABEL_MAP.items()}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_data(path: str) -> pd.DataFrame:
    """Load the airline tweets CSV file."""
    df = pd.read_csv(path)
    
    # Check for required columns
    required = ["airline_sentiment", "text"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Need: {required}")
    
    # Keep only what we need, drop nulls
    df = df[required].dropna()
    
    print(f"✓ Loaded {len(df)} tweets from {path}")
    return df


# Regex patterns for cleaning
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#")


def clean_text(text: str) -> str:
    """
    Clean tweet text.
    
    - Lowercase
    - Remove URLs and @mentions  
    - Remove hashtag symbols (keep words)
    - Remove special characters (keep letters and spaces)
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def preprocess_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataframe.
    
    Operations:
    - Keep text and sentiment columns only
    - Drop rows with missing values
    - Clean text
    - Create numeric labels (0=neg, 1=neutral, 2=pos)
    """
    df = df_raw.copy()
    
    df = df[["text", "airline_sentiment"]]
    df = df.dropna(subset=["text", "airline_sentiment"])
    
    # Only valid sentiments
    df = df[df["airline_sentiment"].isin(INT_FROM_STR.keys())]
    
    # Clean text and map labels
    df["clean_text"] = df["text"].apply(clean_text)
    df["label"] = df["airline_sentiment"].map(INT_FROM_STR)
    
    # Show distribution
    label_dist = df["label"].value_counts().sort_index()
    print("\n✓ Preprocessed - Label distribution:")
    for label_id, count in label_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {LABEL_MAP[label_id]:10s}: {count:5d} ({pct:5.1f}%)")
    
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split into train/val/test with stratification.
    
    Uses stratified sampling to keep class distribution consistent across splits.
    """
    X = df["clean_text"]
    y = df["label"]
    
    # First split: test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: val and test from remaining
    val_adjusted = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    total = len(X)
    print(f"\n✓ Data split:")
    print(f"  Train: {len(X_train):5d} ({len(X_train)/total*100:5.1f}%)")
    print(f"  Val:   {len(X_val):5d} ({len(X_val)/total*100:5.1f}%)")
    print(f"  Test:  {len(X_test):5d} ({len(X_test)/total*100:5.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_and_train(
    X_train: pd.Series,
    y_train: pd.Series,
    max_features: int = 20000,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Vectorize with TF-IDF and train Logistic Regression.
    
    TF-IDF config:
    - 20K features (vocabulary size)
    - Unigrams + bigrams 
    - min_df=5 (ignore rare words)
    
    LogReg config:
    - balanced class weights (handles imbalance)
    - 1000 iterations
    """
    print("\n✓ Vectorizing...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    
    vocab_sz = len(vectorizer.get_feature_names_out())
    print(f"  Vocab: {vocab_sz} | Shape: {X_train_vec.shape}")
    
    print("✓ Training model...")
    
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)
    
    print("  Done")
    
    return vectorizer, clf


def evaluate_model(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X: pd.Series,
    y_true: pd.Series,
    set_name: str = "Dataset",
) -> Dict:
    """Evaluate model on a dataset."""
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)
    
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    
    report = classification_report(
        y_true, y_pred,
        target_names=[LABEL_MAP[i] for i in range(3)],
        zero_division=0,
    )
    
    print(f"\n{'='*50}")
    print(f"EVALUATION: {set_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nReport:")
    print(report)
    
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def predict_sentiment(
    texts: Union[str, List[str]],
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    label_map: Dict = None,
) -> List[str]:
    """Predict sentiment for text(s)."""
    if label_map is None:
        label_map = LABEL_MAP
    
    if isinstance(texts, str):
        texts = [texts]
    
    X_vec = vectorizer.transform(texts)
    preds = model.predict(X_vec)
    
    return [label_map[int(p)] for p in preds]


def get_prediction_proba(
    text: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    label_map: Dict = None,
) -> Dict[str, float]:
    """Get probability distribution for a text."""
    if label_map is None:
        label_map = LABEL_MAP
    
    X_vec = vectorizer.transform([text])
    probs = model.predict_proba(X_vec)[0]
    
    result = {}
    for cls, prob in zip(model.classes_, probs):
        result[label_map[int(cls)]] = float(prob)
    
    return result


def get_feature_importance(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    top_n: int = 15,
) -> Dict[str, Dict]:
    """Get top N features per class using model coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    
    importance = {}
    for idx, label in LABEL_MAP.items():
        coefs = model.coef_[idx]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        
        importance[label] = {
            feature_names[i]: float(coefs[i])
            for i in top_idx
        }
    
    return importance
