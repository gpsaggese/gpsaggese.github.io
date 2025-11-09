"""
utils_data_preprocessing.py

Utility functions for preprocessing and preparing news text data for fake news detection.

This module provides functions for:
- Text cleaning and normalization
- Tokenization and stopword removal
- Stemming and lemmatization
- Feature extraction (TF-IDF, word embeddings)
- Data loading and validation
"""

import pandas as pd
import numpy as np
import logging
import re
from typing import Tuple, List, Dict, Optional
from pathlib import Path

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ============================================================================
# Text Cleaning and Preprocessing
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing special characters, URLs, and extra whitespace.

    :param text: raw text to clean
    :return: cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and digits (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into individual words.

    :param text: cleaned text to tokenize
    :return: list of tokens
    """
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords from token list.

    :param tokens: list of tokens
    :return: filtered list of tokens
    """
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply stemming to reduce tokens to their root form.

    :param tokens: list of tokens
    :return: stemmed tokens
    """
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to reduce tokens to their base form.

    :param tokens: list of tokens
    :return: lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str, use_stemming: bool = True) -> str:
    """
    Complete preprocessing pipeline: clean, tokenize, remove stopwords, and stem/lemmatize.

    :param text: raw text to preprocess
    :param use_stemming: if True use stemming, else use lemmatization
    :return: preprocessed text as a single string
    """
    # Clean
    text = clean_text(text)

    # Tokenize
    tokens = tokenize_text(text)

    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Stem or Lemmatize
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    return ' '.join(tokens)


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_tfidf_features(
    texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    fit_on_data: Optional[List[str]] = None
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Extract TF-IDF features from text documents.

    :param texts: list of text documents
    :param max_features: maximum number of features to extract
    :param ngram_range: range of n-grams to consider
    :param fit_on_data: if provided, fit vectorizer on this data instead
    :return: TF-IDF feature matrix and fitted vectorizer
    """
    logger.info(f"Extracting TF-IDF features with max_features={max_features}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8
    )

    if fit_on_data is not None:
        logger.info("Fitting vectorizer on provided training data")
        vectorizer.fit(fit_on_data)
        features = vectorizer.transform(texts).toarray()
    else:
        features = vectorizer.fit_transform(texts).toarray()

    logger.info(f"TF-IDF matrix shape: {features.shape}")
    return features, vectorizer


# ============================================================================
# Data Loading and Splitting
# ============================================================================


def load_fake_news_data(file_path: str) -> pd.DataFrame:
    """
    Load fake news dataset from CSV file.

    Expected columns: id, title, author, text, label (0=fake, 1=real)

    :param file_path: path to CSV file
    :return: pandas DataFrame
    """
    logger.info(f"Loading dataset from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Check for required columns
        expected_cols = {'title', 'text', 'label'}
        if not expected_cols.issubset(set(df.columns)):
            logger.warning(f"Dataset missing expected columns. Found: {df.columns.tolist()}")

        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that dataset has required columns and sufficient data.

    :param df: pandas DataFrame
    :return: True if valid, raises exception otherwise
    """
    if df.empty:
        raise ValueError("Dataset is empty")

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Check for missing values
    missing_text = df['text'].isna().sum()
    missing_label = df['label'].isna().sum()

    logger.info(f"Missing values - text: {missing_text}, label: {missing_label}")

    if missing_text > 0 or missing_label > 0:
        logger.warning("Dataset contains missing values. These will be removed.")
        df = df.dropna(subset=['text', 'label'])

    return True


def prepare_dataset(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    preprocessing: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare dataset by preprocessing text and splitting into train/test sets.

    :param df: input DataFrame
    :param text_column: name of text column
    :param label_column: name of label column
    :param preprocessing: if True, preprocess text data
    :param test_size: proportion of test data
    :param random_state: random seed for reproducibility
    :return: (X_train, X_test, y_train, y_test) where X contains preprocessed text
    """
    logger.info("Preparing dataset...")

    # Validate data
    validate_data(df)

    # Create a copy to avoid modifying original
    df = df.copy()

    # Remove rows with missing values in text or label
    df = df.dropna(subset=[text_column, label_column])

    # Preprocess text if requested
    if preprocessing:
        logger.info("Preprocessing text data...")
        df[text_column] = df[text_column].apply(preprocess_text)

    # Separate features and target
    X = df[[text_column]].copy()
    X = X.rename(columns={text_column: 'text'})
    y = df[label_column].copy()

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )

    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    logger.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution in test: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def get_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate basic statistics about text documents.

    :param texts: list of text documents
    :return: dictionary with statistics
    """
    lengths = [len(text.split()) for text in texts]

    stats = {
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths)
    }

    logger.info(f"Text statistics: {stats}")
    return stats
