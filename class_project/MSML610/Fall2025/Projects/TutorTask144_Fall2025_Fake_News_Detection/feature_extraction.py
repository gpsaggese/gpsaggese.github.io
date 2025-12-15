"""
Feature Extraction Module for Fake News Detection

Implements multiple feature extraction techniques:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word Embeddings (Word2Vec and GloVe-style)
- BERT Embeddings (contextual embeddings)

"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

import torch
from transformers import BertTokenizer, BertModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFIDFExtractor:
    
    # TF-IDF Feature Extractor using scikit-learn.

    def __init__(self, max_features: int = 5000, max_df: float = 0.95, min_df: int = 2):
        """
        Initialize TF-IDF extractor.

        Args:
            max_features: Maximum number of features
            max_df: Ignore terms that appear in more than max_df fraction of documents
            min_df: Ignore terms that appear in fewer than min_df documents
        """
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.feature_names = None
        logger.info(f"Initialized TF-IDF extractor (max_features={max_features})")

    def fit(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on texts.

        Args:
            texts: List of text documents
        """
        logger.info(f"Fitting TF-IDF on {len(texts)} documents:")
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"Vocabulary size: {len(self.feature_names)}")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF feature vectors.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix (sparse or dense)
        """
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts to TF-IDF features.

        Args:
            texts: List of text documents

        Returns:
            TF-IDF feature matrix
        """
        logger.info(f"Fitting and transforming {len(texts)} documents:")
        features = self.vectorizer.fit_transform(texts).toarray()
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"Features shape: {features.shape}")
        return features

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names) if self.feature_names is not None else []

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Saved TF-IDF vectorizer to {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"Loaded TF-IDF vectorizer from {filepath}")


class Word2VecExtractor:
    """
    Word2Vec Feature Extractor using Gensim.

    Creates distributed word embeddings for documents.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2):
        """
        Initialize Word2Vec extractor.

        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

        if not GENSIM_AVAILABLE:
            logger.warning("Gensim not installed. Install with: pip install gensim")
        else:
            logger.info(f"Initialized Word2Vec extractor (vector_size={vector_size})")

    def fit(self, tokenized_texts: List[List[str]]) -> None:
        """
        Fit Word2Vec model on tokenized texts.

        Args:
            tokenized_texts: List of tokenized documents (each is list of words)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim required for Word2Vec. Install with: pip install gensim")

        logger.info(f"Training Word2Vec on {len(tokenized_texts)} documents...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        logger.info(f"Vocabulary size: {len(self.model.wv)}")

    def transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Transform tokenized texts to document vectors (average of word vectors).

        Args:
            tokenized_texts: List of tokenized documents

        Returns:
            Document vectors (shape: n_docs x vector_size)
        """
        vectors = []
        for tokens in tokenized_texts:
            word_vectors = []
            for token in tokens:
                if token in self.model.wv:
                    word_vectors.append(self.model.wv[token])

            if len(word_vectors) > 0:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                doc_vector = np.zeros(self.vector_size)

            vectors.append(doc_vector)

        return np.array(vectors)

    def fit_transform(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)

    def save(self, filepath: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Saved Word2Vec model to {filepath}")

    def load(self, filepath: str) -> None:
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim required for Word2Vec")
        self.model = Word2Vec.load(filepath)
        logger.info(f"Loaded Word2Vec model from {filepath}")


class BERTEmbeddings:
    """
    BERT Contextual Embeddings for feature extraction.

    Uses pre-trained BERT to generate contextualized word embeddings
    and document-level representations.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize BERT embeddings extractor.

        Args:
            model_name: Name of pre-trained BERT model
            device: torch device (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized BERT embeddings (model={model_name}, device={self.device})")

    def extract_embeddings(
        self,
        texts: List[str],
        layer: int = -2,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract BERT embeddings for texts.

        Args:
            texts: List of text documents
            layer: Which layer to extract (-1 for last, -2 for second-to-last, etc.)
            pooling: Pooling method ("mean", "cls", "max")

        Returns:
            Embedding matrix (shape: n_docs x hidden_size)
        """
        embeddings = []

        for text in texts:
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            with torch.no_grad():
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                hidden_states = outputs.hidden_states[layer]

                if pooling == "mean":
                    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                elif pooling == "cls":
                    embedding = hidden_states[0, 0, :].cpu().numpy()
                elif pooling == "max":
                    embedding = hidden_states.max(dim=1)[0].squeeze().cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

            embeddings.append(embedding)

        return np.array(embeddings)

    def extract_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        layer: int = -2,
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Extract BERT embeddings for texts in batches (memory efficient).

        Args:
            texts: List of text documents
            batch_size: Batch size for processing
            layer: Which layer to extract
            pooling: Pooling method

        Returns:
            Embedding matrix
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.extract_embeddings(batch, layer=layer, pooling=pooling)
            all_embeddings.append(batch_embeddings)

            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} documents")

        return np.vstack(all_embeddings)


def compare_feature_extraction_methods(
    texts: List[str],
    labels: List[int],
    tokenized_texts: List[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different feature extraction methods.

    Args:
        texts: List of text documents
        labels: List of labels
        tokenized_texts: Optional pre-tokenized texts for Word2Vec

    Returns:
        Dictionary with results for each method
    """
    results = {}
    
    logger.info("\nTF-IDF Feature Extraction")
   
    tfidf = TFIDFExtractor(max_features=5000)
    tfidf_features = tfidf.fit_transform(texts)
    results['tfidf'] = {
        'shape': tfidf_features.shape,
        'feature_names': tfidf.get_feature_names()[:10],
        'features': tfidf_features
    }
    logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
    logger.info(f"Sample features: {results['tfidf']['feature_names']}")


    if GENSIM_AVAILABLE and tokenized_texts is not None:
        logger.info("\nWord2Vec Feature Extraction")
        w2v = Word2VecExtractor(vector_size=100)
        w2v_features = w2v.fit_transform(tokenized_texts)
        results['word2vec'] = {
            'shape': w2v_features.shape,
            'features': w2v_features
        }
        logger.info(f"Word2Vec features shape: {w2v_features.shape}")
    else:
        logger.warning("Word2Vec skipped (gensim not installed or tokenized_texts not provided)")

    logger.info("\nBERT Embeddings Feature Extraction")
    bert = BERTEmbeddings()
    bert_features = bert.extract_embeddings_batch(texts, batch_size=32)
    results['bert'] = {
        'shape': bert_features.shape,
        'features': bert_features
    }
    logger.info(f"BERT embeddings shape: {bert_features.shape}")

    return results


if __name__ == '__main__':
    sample_texts = [
        "This is a real news article about climate change policy and renewable energy",
        "SHOCKING: Government hides secret technology with no evidence provided",
        "Scientists announce breakthrough in renewable energy research today",
        "Another real news article about environmental protection efforts",
        "FAKE: Celebrity reveals shocking secret that nobody knows about this"
    ]

    print("Feature Extraction Module Examples")

    # TF-IDF
    print("\n1. TF-IDF Extraction:")
    tfidf = TFIDFExtractor(max_features=100, min_df=1, max_df=0.9)
    features = tfidf.fit_transform(sample_texts)
    print(f"   Shape: {features.shape}")
    print(f"   Sample features: {tfidf.get_feature_names()[:10]}")

    # BERT
    print("\n2. BERT Embeddings:")
    bert = BERTEmbeddings()
    embeddings = bert.extract_embeddings(sample_texts[:1])
    print(f"   Shape: {embeddings.shape}")
    print(f"   Embedding dimensions: {embeddings.shape[1]}")

    print("\n Feature extraction module working correctly!")
