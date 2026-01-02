"""
Semantic Search Utilities Module
Provides core functionality for semantic search over text documents.
"""

from dataclasses import dataclass
from typing import List, Tuple, Protocol
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    """Container for a single search result."""
    index: int
    score: float
    text: str


class EmbeddingModel(Protocol):
    """Protocol for sentence embedding models."""
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        """Encode sentences into embeddings."""
        ...


class SemanticSearchEngine:
    """
    Semantic search engine using sentence transformers.
    
    This is the core API - it defines the interface for semantic search
    without specifying implementation details.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        self.model_name = model_name
        self.model: SentenceTransformer = None
        self.embeddings: np.ndarray = None
        self.texts: List[str] = None
    
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        self.model = SentenceTransformer(self.model_name)
    
    def index_documents(self, texts: List[str]) -> None:
        """
        Index a collection of documents.
        
        Args:
            texts: List of text documents to index
        """
        if self.model is None:
            self.load_model()
        
        self.texts = texts
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=256,
            convert_to_numpy=True
        )
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        if self.embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                index=int(idx),
                score=float(similarities[idx]),
                text=self.texts[idx]
            ))
        
        return results


def load_wikipedia_data(file_path: str, max_articles: int = None) -> List[str]:
    """
    Load Wikipedia articles from parquet file.
    
    Args:
        file_path: Path to parquet file
        max_articles: Maximum number of articles to load
        
    Returns:
        List of article texts
    """
    import pandas as pd
    
    df = pd.read_parquet(file_path, columns=["text"])
    df = df.dropna(subset=["text"])
    
    texts = df["text"].tolist()
    
    if max_articles:
        texts = texts[:max_articles]
    
    return texts