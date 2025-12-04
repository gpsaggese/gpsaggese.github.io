import re
from typing import Tuple, Dict

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

# ----- Label maps -----------------------------------------------------------

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ----- Data loading & cleaning ---------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load the Twitter US Airline Sentiment dataset and keep only text + label.
    """
    df = pd.read_csv(path)
    # dataset usually has columns 'airline_sentiment' and 'text'
    if "airline_sentiment" not in df.columns or "text" not in df.columns:
        raise ValueError("Expected columns 'airline_sentiment' and 'text' not found.")
    df = df[["text", "airline_sentiment"]].dropna()
    return df


URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]+")


def clean_text(text: str) -> str:
    """
    Basic tweet cleaning:
    - lowercasing
    - remove URLs and @mentions
    - drop '#' symbol (keep the word)
    - keep only letters and spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning and map labels to integers.
    Output columns:
        - text (original)
        - clean_text
        - label (int)
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    df["label"] = df["airline_sentiment"].map(LABEL_MAP)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df


# ----- Split & vectorize ---------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split into train / val / test on clean_text + label.
    """
    X = df["clean_text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # from the remaining temp set, carve out validation
    val_rel_size = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_rel_size,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_and_train(
    X_train: pd.Series,
    y_train: pd.Series,
    max_features: int = 20000,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Fit TF-IDF vectorizer and a Logistic Regression classifier.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train_vec, y_train)
    return vectorizer, clf


# ----- Evaluation ----------------------------------------------------------

def evaluate_model(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X: pd.Series,
    y_true: pd.Series,
) -> Dict:
    """
    Transform X, predict, and compute metrics.
    Returns a dict with accuracy, macro precision/recall/F1,
    confusion matrix, and a text classification report.
    """
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP.keys())],
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
