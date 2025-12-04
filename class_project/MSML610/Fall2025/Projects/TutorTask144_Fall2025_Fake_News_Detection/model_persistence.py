"""
Model Persistence Module

Handles saving and loading of trained models to avoid retraining.
Supports BERT and LSTM models with metadata storage.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Manages model saving and loading for trained deep learning models."""

    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model persistence manager.

        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load model registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'models': {}}

    def _save_registry(self):
        """Save model registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def save_bert_model(
        self,
        model,
        tokenizer,
        model_name: str,
        metrics: Dict,
        training_config: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save BERT model, tokenizer, and metadata.

        Args:
            model: Trained BERT model
            tokenizer: BERT tokenizer
            model_name: Name for the saved model
            metrics: Model performance metrics
            training_config: Training configuration used
            metadata: Additional metadata

        Returns:
            Model ID (directory name)
        """
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            model.save_pretrained(str(model_dir / "model"))
            logger.info(f"✓ Saved BERT model to {model_dir / 'model'}")

            # Save tokenizer
            tokenizer.save_pretrained(str(model_dir / "tokenizer"))
            logger.info(f"✓ Saved tokenizer to {model_dir / 'tokenizer'}")

            # Save metadata
            metadata_dict = {
                'model_type': 'bert',
                'model_name': model_name,
                'model_id': model_id,
                'created_at': timestamp,
                'metrics': metrics,
                'training_config': training_config,
                'custom_metadata': metadata or {},
                'saved_state': 'complete'
            }

            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            logger.info(f"✓ Saved metadata to {model_dir / 'metadata.json'}")

            # Update registry
            self.registry['models'][model_id] = {
                'type': 'bert',
                'path': str(model_dir),
                'created_at': timestamp,
                'accuracy': metrics.get('accuracy', 0),
                'f1': metrics.get('f1', 0)
            }
            self._save_registry()

            logger.info(f"✓ Model saved successfully as: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"✗ Failed to save model: {e}")
            raise

    def load_bert_model(
        self,
        model_id: str,
        device: str = 'cpu'
    ):
        """
        Load BERT model and tokenizer.

        Args:
            model_id: Model ID to load
            device: Device to load model to ('cpu' or 'cuda')

        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_id} not found")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir / "model")
            ).to(device)
            logger.info(f"✓ Loaded BERT model from {model_dir / 'model'}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
            logger.info(f"✓ Loaded tokenizer from {model_dir / 'tokenizer'}")

            # Load metadata
            with open(model_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            logger.info(f"✓ Loaded metadata")

            return model, tokenizer, metadata

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def list_models(self) -> Dict:
        """
        List all saved models with their metadata.

        Returns:
            Dictionary of models and their info
        """
        models_list = {}
        for model_id, info in self.registry.get('models', {}).items():
            models_list[model_id] = {
                'type': info.get('type'),
                'created_at': info.get('created_at'),
                'accuracy': info.get('accuracy'),
                'f1': info.get('f1'),
                'exists': Path(info.get('path', '')).exists()
            }
        return models_list

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a saved model.

        Args:
            model_id: Model ID to delete

        Returns:
            True if deleted, False if not found
        """
        model_dir = self.models_dir / model_id

        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            logger.info(f"✓ Deleted model directory: {model_dir}")

            # Remove from registry
            if model_id in self.registry['models']:
                del self.registry['models'][model_id]
                self._save_registry()
                logger.info(f"✓ Removed model from registry")

            return True
        else:
            logger.warning(f"✗ Model {model_id} not found")
            return False

    def get_best_model(self, metric: str = 'accuracy') -> Optional[str]:
        """
        Get the best model based on a metric.

        Args:
            metric: Metric to sort by ('accuracy', 'f1')

        Returns:
            Best model ID or None
        """
        if not self.registry['models']:
            return None

        best_model = max(
            self.registry['models'].items(),
            key=lambda x: x[1].get(metric, 0)
        )
        return best_model[0]

    def print_registry(self):
        """Print model registry in a readable format."""
        print("\n" + "="*80)
        print("MODEL REGISTRY")
        print("="*80)

        models = self.list_models()
        if not models:
            print("No models saved yet.")
            return

        for model_id, info in models.items():
            print(f"\nModel ID: {model_id}")
            print(f"  Type:       {info['type']}")
            print(f"  Created:    {info['created_at']}")
            print(f"  Accuracy:   {info['accuracy']:.4f}")
            print(f"  F1:         {info['f1']:.4f}")
            print(f"  Exists:     {'Yes' if info['exists'] else 'No'}")

        print("\n" + "="*80 + "\n")
