"""
Data Preprocessing Module for Fake News Detection

Implements text preprocessing:
- Tokenization
- Stopword removal
- Stemming/Lemmatization
- Normalization
"""

import re
import logging
from typing import List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or len(text) == 0:
        return []

    try:
        tokens = word_tokenize(text)
        return [token for token in tokens if token.isalnum()]
    except Exception as e:
        logger.warning(f"Tokenization error: {e}")
        return text.split()


def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    try:
        stop_words = set(stopwords.words(language))
    except Exception as e:
        logger.warning(f"Could not load stopwords: {e}. Skipping stopword removal.")
        return tokens

    filtered = [token for token in tokens if token.lower() not in stop_words]
    return filtered


def stem_tokens(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed


def preprocess_pipeline(
    text: str,
    normalize: bool = True,
    tokenize_text: bool = True,
    remove_stops: bool = True,
    stem: bool = True
) -> str:
    if normalize:
        text = normalize_text(text)

    if tokenize_text:
        tokens = tokenize(text)
    else:
        tokens = text.split()

    if remove_stops:
        tokens = remove_stopwords(tokens)

    if stem:
        tokens = stem_tokens(tokens)
        
    return ' '.join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'content',
    normalize: bool = True,
    tokenize_text: bool = True,
    remove_stops: bool = True,
    stem: bool = True
) -> pd.DataFrame:
    
    logger.info(f"Preprocessing {len(df)} texts...")

    df_copy = df.copy()

    # Apply preprocessing pipeline to each text
    df_copy[text_column] = df_copy[text_column].apply(
        lambda x: preprocess_pipeline(
            x,
            normalize=normalize,
            tokenize_text=tokenize_text,
            remove_stops=remove_stops,
            stem=stem
        )
    )

    logger.info(f"Preprocessing complete. Removed {(df_copy[text_column] == '').sum()} empty texts.")

    df_copy = df_copy[df_copy[text_column].str.len() > 0]

    return df_copy


def get_preprocessing_stats(original_df: pd.DataFrame, preprocessed_df: pd.DataFrame, text_column: str = 'content') -> dict:
    
    original_lengths = original_df[text_column].apply(len)
    preprocessed_lengths = preprocessed_df[text_column].apply(len)

    return {
        'original_samples': len(original_df),
        'preprocessed_samples': len(preprocessed_df),
        'samples_removed': len(original_df) - len(preprocessed_df),
        'avg_original_length': original_lengths.mean(),
        'avg_preprocessed_length': preprocessed_lengths.mean(),
        'length_reduction_pct': 100 * (1 - preprocessed_lengths.mean() / original_lengths.mean())
    }


# Example usage
if __name__ == '__main__':
    # Test preprocessing on sample texts
    sample_texts = [
        "This is a REAL news article from Reuters about climate change.",
        "SHOCKING: Government hides SECRET technology! No evidence provided.",
        "The hospital announced a major expansion http://example.com contact: news@hospital.org"
    ]

    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")

    print("\nPreprocessing results:")

    for i, text in enumerate(sample_texts, 1):
        preprocessed = preprocess_pipeline(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Preprocessed: {preprocessed}")
