"""
Category-Based Contextual Adaptation for Fake News Detection.

Implements category-aware model selection and prediction routing based on
article category or source type for improved accuracy.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("category_adaptation")


class CategoryDetector:
    """Detects article category from text."""

    # Category keywords
    CATEGORY_KEYWORDS = {
        'politics': ['election', 'vote', 'congress', 'senate', 'president', 'campaign', 'political', 'law', 'bill', 'democratic', 'republican'],
        'health': ['health', 'medical', 'doctor', 'disease', 'virus', 'vaccine', 'hospital', 'treatment', 'patient', 'covid'],
        'business': ['business', 'market', 'stock', 'company', 'economy', 'trade', 'investment', 'financial', 'profit', 'sales'],
        'science': ['science', 'study', 'research', 'scientist', 'discovery', 'physics', 'chemistry', 'biology', 'experiment', 'data'],
        'entertainment': ['movie', 'actor', 'celebrity', 'film', 'music', 'artist', 'entertainment', 'hollywood', 'show', 'performance'],
        'sports': ['sport', 'game', 'team', 'player', 'coach', 'league', 'football', 'basketball', 'match', 'championship'],
        'other': []
    }

    @staticmethod
    def detect_category(text: str) -> str:
        """
        Detect article category from text.

        Args:
            text: Article text

        Returns:
            Detected category
        """
        text_lower = text.lower()
        category_scores = {}

        for category, keywords in CategoryDetector.CATEGORY_KEYWORDS.items():
            if category == 'other':
                continue
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score

        if not category_scores:
            return 'other'

        return max(category_scores, key=category_scores.get)

    @staticmethod
    def get_all_categories() -> List[str]:
        """Get list of all categories."""
        return list(CategoryDetector.CATEGORY_KEYWORDS.keys())


class CategoryBasedAdapter:
    """Adapts model predictions based on article category."""

    def __init__(self):
        """Initialize adapter."""
        self.category_models = {}  # Store category-specific models
        self.category_thresholds = {}  # Category-specific decision thresholds
        self.category_stats = {}  # Statistics per category

    def register_category_model(
        self,
        category: str,
        model,
        threshold: float = 0.5,
        stats: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Register a model for specific category.

        Args:
            category: Category name
            model: Trained model instance
            threshold: Decision threshold for this category
            stats: Category-specific statistics
        """
        self.category_models[category] = model
        self.category_thresholds[category] = threshold
        if stats:
            self.category_stats[category] = stats
        logger.info(f"Registered model for category: {category} (threshold: {threshold})")

    def predict_with_adaptation(
        self,
        text: str,
        default_model,
        use_category_specific: bool = True
    ) -> Tuple[int, str, Dict[str, Any]]:
        """
        Make prediction with category adaptation.

        Args:
            text: Article text
            default_model: Default model to use if no category-specific model
            use_category_specific: Whether to use category-specific models

        Returns:
            Tuple of (prediction, category, metadata)
        """
        category = CategoryDetector.detect_category(text)

        metadata = {
            'category': category,
            'model_used': 'default',
            'threshold': 0.5
        }

        # Use category-specific model if available
        if use_category_specific and category in self.category_models:
            model = self.category_models[category]
            threshold = self.category_thresholds.get(category, 0.5)
            metadata['model_used'] = category
            metadata['threshold'] = threshold
            prediction = model.predict(text)
        else:
            model = default_model
            prediction = model.predict(text)

        return prediction, category, metadata

    def batch_predict_with_adaptation(
        self,
        texts: List[str],
        default_model,
        use_category_specific: bool = True
    ) -> List[Tuple[int, str, Dict[str, Any]]]:
        """
        Make batch predictions with category adaptation.

        Args:
            texts: List of article texts
            default_model: Default model
            use_category_specific: Whether to use category-specific models

        Returns:
            List of (prediction, category, metadata) tuples
        """
        results = []
        for text in texts:
            result = self.predict_with_adaptation(text, default_model, use_category_specific)
            results.append(result)
        return results

    def get_category_distribution(self, texts: List[str]) -> Dict[str, int]:
        """Get distribution of categories in texts."""
        distribution = {cat: 0 for cat in CategoryDetector.get_all_categories()}

        for text in texts:
            category = CategoryDetector.detect_category(text)
            distribution[category] += 1

        return distribution

    def get_category_performance(self, category: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for category."""
        return self.category_stats.get(category)

    def print_category_info(self) -> None:
        """Print information about registered categories."""
        print("\n" + "="*80)
        print("CATEGORY-BASED ADAPTATION CONFIGURATION")
        print("="*80)

        if not self.category_models:
            print("No category-specific models registered")
            return

        for category in self.category_models:
            print(f"\nCategory: {category.upper()}")
            print(f"  Threshold: {self.category_thresholds.get(category, 'N/A')}")

            stats = self.category_stats.get(category)
            if stats:
                print(f"  Performance:")
                for metric, value in stats.items():
                    print(f"    {metric}: {value:.4f}")

        print("\n" + "="*80)


class ContextAwarePredictor:
    """Combines MCP context with category adaptation."""

    def __init__(self, adapter: CategoryBasedAdapter):
        """Initialize predictor."""
        self.adapter = adapter

    def predict_with_context(
        self,
        text: str,
        default_model,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with full context.

        Args:
            text: Article text
            default_model: Default model
            return_confidence: Include confidence scores

        Returns:
            Dictionary with prediction and context
        """
        prediction, category, metadata = self.adapter.predict_with_adaptation(
            text, default_model, use_category_specific=True
        )

        result = {
            'prediction': prediction,
            'label': 'Fake' if prediction == 1 else 'Real',
            'category': category,
            'model_used': metadata['model_used'],
            'decision_threshold': metadata['threshold'],
            'text_preview': text[:100] + "..." if len(text) > 100 else text
        }

        return result


def analyze_fake_news_by_category(
    texts: List[str],
    labels: List[int],
    dataset_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Analyze fake news distribution by category.

    Args:
        texts: List of article texts
        labels: List of labels (0=real, 1=fake)
        dataset_name: Name of dataset

    Returns:
        Analysis results
    """
    detector = CategoryDetector()
    analysis = {
        'dataset': dataset_name,
        'total_articles': len(texts),
        'categories': {},
        'category_distribution': detector.get_category_distribution(texts)
    }

    for category in detector.get_all_categories():
        cat_texts = [t for t, label in zip(texts, labels)]
        cat_labels = [label for cat, label in zip(
            [detector.detect_category(t) for t in texts],
            labels
        ) if cat == category]

        if cat_labels:
            fake_count = sum(cat_labels)
            total = len(cat_labels)
            analysis['categories'][category] = {
                'total': total,
                'fake': fake_count,
                'real': total - fake_count,
                'fake_percentage': (fake_count / total * 100) if total > 0 else 0
            }

    return analysis
