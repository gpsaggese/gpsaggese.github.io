"""
Fake News Detection Utils
Simple ML pipeline for classifying news articles as real or fake.
Uses TF-IDF + PassiveAggressiveClassifier.
"""

import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_dataset(data_dir="data"):
    """
    Load base dataset from data/true.csv and data/fake.csv.

    Expected columns: title, text, subject, date

    Returns:
        pd.DataFrame with columns: content, label
        - content: combined title + text
        - label: 1 for real (true.csv), 0 for fake (fake.csv)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    real_csv = data_path / "true.csv"
    if not real_csv.exists():
        raise FileNotFoundError(f"File not found: {real_csv}")

    true_df = pd.read_csv(real_csv)
    logger.info(f"Loaded {len(true_df)} real articles from {real_csv}")

    fake_csv = data_path / "fake.csv"
    if not fake_csv.exists():
        raise FileNotFoundError(f"File not found: {fake_csv}")

    fake_df = pd.read_csv(fake_csv)
    logger.info(f"Loaded {len(fake_df)} fake articles from {fake_csv}")

    dfs = []
    
    if 'title' in true_df.columns and 'text' in true_df.columns:
        true_df['content'] = true_df['title'].fillna('') + "\n\n" + true_df['text'].fillna('')
        true_df['label'] = 1
        dfs.append(true_df[['content', 'label']])

    if 'title' in fake_df.columns and 'text' in fake_df.columns:
        fake_df['content'] = fake_df['title'].fillna('') + "\n\n" + fake_df['text'].fillna('')
        fake_df['label'] = 0
        dfs.append(fake_df[['content', 'label']])

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[combined_df['content'].str.strip() != '']
    combined_df = combined_df.dropna(subset=['content', 'label'])

    logger.info(f"Combined dataset: {len(combined_df)} samples")
    logger.info(f"  Real: {(combined_df['label'] == 1).sum()}")
    logger.info(f"  Fake: {(combined_df['label'] == 0).sum()}")

    return combined_df


def load_news_full_dataset(data_dir="data"):
    """
    Load extended dataset if available (handles various formats).

    Looks for additional CSV files with different naming conventions:
    - fake.csv / real.csv
    - Fake.csv / True.csv (case variations)
    - Files with 'fake', 'real', 'true' in the name

    Returns:
        pd.DataFrame with columns: content, label
        - label: 1 for real, 0 for fake
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return None

    dfs = []

    for csv_file in data_path.glob("*.csv"):
        filename = csv_file.stem.lower()

        try:
            df = pd.read_csv(csv_file)
            if 'fake' in filename:
                label = 0
            elif 'real' in filename or 'true' in filename:
                label = 1
            else:
                continue

            content_cols = []
            for col in df.columns:
                if col.lower() in ['title', 'text', 'content', 'article']:
                    content_cols.append(col)

            if len(content_cols) == 0:
                logger.warning(f"No content columns found in {csv_file}, skipping")
                continue

            df['content'] = df[content_cols[0]].fillna('')
            for col in content_cols[1:]:
                df['content'] = df['content'] + "\n\n" + df[col].fillna('')

            df['label'] = label
            dfs.append(df[['content', 'label']])
            logger.info(f"Loaded {len(df)} articles from {csv_file}")

        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")
            continue

    if not dfs:
        logger.warning("No datasets loaded")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df[combined_df['content'].str.strip() != '']
    combined_df = combined_df.dropna(subset=['content', 'label'])

    logger.info(f"Full dataset: {len(combined_df)} samples")
    logger.info(f"  Real: {(combined_df['label'] == 1).sum()}")
    logger.info(f"  Fake: {(combined_df['label'] == 0).sum()}")

    return combined_df


def clean_text(text):
    """
    Clean and normalize text.
    - Lowercase
    - Remove special characters (keep spaces and alphanumeric)
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(csv_path):
    # Load data from CSV with columns: text, label.
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from {csv_path}")

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")

        df = df.dropna(subset=['text', 'label'])
        logger.info(f"After cleaning: {len(df)} samples")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def extract_features(texts, vectorizer=None, fit=False):
    """
    Convert texts to TF-IDF vectors.

    Args:
        texts: list or array of text strings
        vectorizer: existing TfidfVectorizer (for test set), optional
        fit: if True, fit vectorizer on these texts

    Returns:
        (vectors, vectorizer)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        if fit:
            vectors = vectorizer.fit_transform(texts)
            logger.info(f"Extracted {vectors.shape[1]} features")
            return vectors, vectorizer

    vectors = vectorizer.transform(texts)
    return vectors, vectorizer


def train_model(X_train, y_train):
    """
    Train PassiveAggressiveClassifier.

    Args:
        X_train: TF-IDF vectors
        y_train: labels (0=fake, 1=real)

    Returns:
        trained model
    """
    model = PassiveAggressiveClassifier(
        C=1.0,
        loss='hinge',
        max_iter=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return metrics.

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix
    """
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred,
        'ground_truth': y_test
    }

    logger.info(f"Accuracy: {results['accuracy']:.3f}")
    logger.info(f"Precision: {results['precision']:.3f}")
    logger.info(f"Recall: {results['recall']:.3f}")
    logger.info(f"F1 Score: {results['f1']:.3f}")

    return results


def predict_text(text, model, vectorizer):
    """
    Predict label for a single text.

    Returns:
        (prediction, confidence)
        prediction: 0 (fake) or 1 (real)
        confidence: float 0.0-1.0
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    decision_score = model.decision_function(vector)[0]
    confidence = 1.0 / (1.0 + np.exp(-decision_score))

    return prediction, confidence


def save_artifacts(model, vectorizer, artifact_dir='artifacts'):
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(exist_ok=True)

    model_path = artifact_path / 'model.pkl'
    vectorizer_path = artifact_path / 'vectorizer.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    logger.info(f"Saved artifacts to {artifact_dir}/")


def load_artifacts(artifact_dir='artifacts'):
    artifact_path = Path(artifact_dir)

    model_path = artifact_path / 'model.pkl'
    vectorizer_path = artifact_path / 'vectorizer.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    logger.info(f"Loaded artifacts from {artifact_dir}/")
    return model, vectorizer


def full_pipeline(train_csv, test_csv, artifact_dir='artifacts'):

    logger.info("FAKE NEWS DETECTION PIPELINE")
    logger.info("=" * 60)

    # Load data
    logger.info("\n[1] Loading data...")
    train_df = load_data(train_csv)
    test_df = load_data(test_csv)

    # Clean texts
    logger.info("\n[2] Cleaning texts...")
    X_train_clean = [clean_text(t) for t in train_df['text']]
    X_test_clean = [clean_text(t) for t in test_df['text']]
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # Extract features
    logger.info("\n[3] Extracting TF-IDF features...")
    X_train_vectors, vectorizer = extract_features(X_train_clean, fit=True)
    X_test_vectors, _ = extract_features(X_test_clean, vectorizer=vectorizer, fit=False)

    # Train model
    logger.info("\n[4] Training classifier...")
    model = train_model(X_train_vectors, y_train)

    # Evaluate
    logger.info("\n[5] Evaluating on test set...")
    results = evaluate_model(model, X_test_vectors, y_test)

    # Save
    logger.info("\n[6] Saving artifacts...")
    save_artifacts(model, vectorizer, artifact_dir)

    return {
        'model': model,
        'vectorizer': vectorizer,
        'results': results,
        'train_df': train_df,
        'test_df': test_df
    }
