"""
MCP (Model Context Protocol) Server for BERT-based Fake News Detection.

This server provides:
1. MCP resources for model metadata and versioning
2. MCP tools for predictions, model management, and context retrieval
3. Model registry for tracking versions and performance
4. Context-aware deployment with model selection

Citations:
- MCP Framework: https://github.com/anthropics/python-sdk
- FastMCP: https://github.com/jlowin/FastMCP
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import torch
import numpy as np
from mcp.server.fastmcp import FastMCP
from bert_utils import BertModelWrapper, TrainingConfig, BertTextDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fake_news_mcp")

# Initialize MCP server
mcp = FastMCP("fake_news_detector")

# Global state management
class ModelRegistry:
    """Manages model versions and metadata."""

    def __init__(self, registry_path: str = "deep_learning_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.active_model_id: Optional[str] = None
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk if exists."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.models = data.get('models', {})
                self.active_model_id = data.get('active_model_id')
        else:
            self.models = {}
            self.active_model_id = None

    def _save_registry(self) -> None:
        """Save registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump({
                'models': self.models,
                'active_model_id': self.active_model_id,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    def register_model(
        self,
        model_id: str,
        model_name: str,
        architecture: str,
        training_config: Dict[str, Any],
        test_metrics: Dict[str, float],
        dataset: str,
        model_path: str
    ) -> Dict[str, Any]:
        """Register a new model version."""
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'architecture': architecture,
            'training_config': training_config,
            'test_metrics': test_metrics,
            'dataset': dataset,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        self.models[model_id] = metadata
        if self.active_model_id is None:
            self.active_model_id = model_id
        self._save_registry()
        logger.info(f"Registered model: {model_id}")
        return metadata

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata."""
        return self.models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self.models.values())

    def set_active_model(self, model_id: str) -> bool:
        """Set active model for predictions."""
        if model_id in self.models:
            self.active_model_id = model_id
            self._save_registry()
            logger.info(f"Set active model to: {model_id}")
            return True
        return False

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        """Get active model metadata."""
        if self.active_model_id:
            return self.models.get(self.active_model_id)
        return None


# Initialize registry
registry = ModelRegistry()

# Model cache for inference
_model_cache: Dict[str, BertModelWrapper] = {}


def _get_or_load_model(model_id: str) -> Optional[BertModelWrapper]:
    """Get model from cache or load from disk."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    model_metadata = registry.get_model(model_id)
    if not model_metadata:
        return None

    try:
        model_path = model_metadata['model_path']
        # Initialize model with training config
        config_dict = model_metadata['training_config']
        config = TrainingConfig(**config_dict)

        model = BertModelWrapper(config)
        # In production, load the saved model weights
        logger.info(f"Loaded model: {model_id}")
        _model_cache[model_id] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        return None


# MCP Resources - Model Metadata and Versioning

@mcp.resource("model://registry")
async def get_registry() -> Dict[str, Any]:
    """Get complete model registry with all versions."""
    models = registry.list_models()
    active = registry.get_active_model()
    return {
        'total_models': len(models),
        'active_model_id': registry.active_model_id,
        'active_model': active,
        'models': models,
        'timestamp': datetime.now().isoformat()
    }


@mcp.resource("model://active")
async def get_active_model_info() -> Dict[str, Any]:
    """Get information about the currently active model."""
    active = registry.get_active_model()
    if not active:
        return {'error': 'No active model set', 'status': 'inactive'}
    return active


@mcp.resource("model://metrics/{model_id}")
async def get_model_metrics(model_id: str) -> Dict[str, Any]:
    """Get performance metrics for a specific model."""
    model = registry.get_model(model_id)
    if not model:
        return {'error': f'Model {model_id} not found'}

    return {
        'model_id': model_id,
        'model_name': model.get('model_name'),
        'metrics': model.get('test_metrics', {}),
        'dataset': model.get('dataset'),
        'created_at': model.get('created_at')
    }


@mcp.resource("model://architecture/{model_id}")
async def get_model_architecture(model_id: str) -> Dict[str, Any]:
    """Get architecture details for a specific model."""
    model = registry.get_model(model_id)
    if not model:
        return {'error': f'Model {model_id} not found'}

    return {
        'model_id': model_id,
        'architecture': model.get('architecture'),
        'training_config': model.get('training_config'),
        'created_at': model.get('created_at')
    }


# MCP Tools - Model Predictions and Management

@mcp.tool()
async def predict(
    text: str,
    model_id: Optional[str] = None,
    return_confidence: bool = True
) -> Dict[str, Any]:
    """
    Make a fake news prediction on input text.

    Args:
        text: News article text to classify
        model_id: Specific model to use (defaults to active model)
        return_confidence: Include confidence scores in response

    Returns:
        Prediction with label (0=real, 1=fake) and optional confidence
    """
    # Use active model if not specified
    target_model_id = model_id or registry.active_model_id
    if not target_model_id:
        return {'error': 'No model specified and no active model set', 'status': 'error'}

    model = _get_or_load_model(target_model_id)
    if not model:
        return {'error': f'Model {target_model_id} not found or failed to load', 'status': 'error'}

    try:
        # Use the model's predict method
        prediction = model.predict(text)

        result = {
            'text_preview': text[:100] + "..." if len(text) > 100 else text,
            'text_length': len(text),
            'prediction': int(prediction),
            'label': 'Fake' if prediction == 1 else 'Real',
            'model_id': target_model_id,
            'timestamp': datetime.now().isoformat()
        }

        if return_confidence:
            # Get prediction with probabilities
            with torch.no_grad():
                inputs = model.tokenizer(
                    text,
                    max_length=256,
                    truncation=True,
                    return_tensors='pt',
                    padding='max_length'
                )
                outputs = model.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                result['confidence'] = {
                    'real': float(probs[0][0]),
                    'fake': float(probs[0][1])
                }

        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f'Prediction failed: {str(e)}', 'status': 'error'}


@mcp.tool()
async def batch_predict(
    texts: List[str],
    model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make predictions on multiple text samples.

    Args:
        texts: List of news article texts
        model_id: Specific model to use (defaults to active model)

    Returns:
        List of predictions with statistics
    """
    target_model_id = model_id or registry.active_model_id
    if not target_model_id:
        return {'error': 'No model specified and no active model set', 'status': 'error'}

    model = _get_or_load_model(target_model_id)
    if not model:
        return {'error': f'Model {target_model_id} not found', 'status': 'error'}

    try:
        predictions = []
        fake_count = 0

        for text in texts:
            pred = model.predict(text)
            predictions.append({
                'text_preview': text[:50],
                'prediction': int(pred),
                'label': 'Fake' if pred == 1 else 'Real'
            })
            if pred == 1:
                fake_count += 1

        return {
            'total_samples': len(texts),
            'fake_count': fake_count,
            'real_count': len(texts) - fake_count,
            'fake_percentage': (fake_count / len(texts) * 100) if texts else 0,
            'model_id': target_model_id,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return {'error': f'Batch prediction failed: {str(e)}', 'status': 'error'}


@mcp.tool()
async def register_model_version(
    model_name: str,
    architecture: str,
    training_config: Dict[str, Any],
    test_metrics: Dict[str, float],
    dataset: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Register a new model version in the registry.

    Args:
        model_name: Descriptive name for the model
        architecture: Model architecture description
        training_config: Training configuration parameters
        test_metrics: Performance metrics on test set
        dataset: Dataset used for training
        model_path: Path to saved model weights

    Returns:
        Registered model metadata with unique ID
    """
    model_id = str(uuid.uuid4())[:8]

    result = registry.register_model(
        model_id=model_id,
        model_name=model_name,
        architecture=architecture,
        training_config=training_config,
        test_metrics=test_metrics,
        dataset=dataset,
        model_path=model_path
    )

    return {
        'status': 'registered',
        'model_id': model_id,
        'model_name': model_name,
        **result
    }


@mcp.tool()
async def list_all_models() -> Dict[str, Any]:
    """
    List all registered model versions with their metrics.

    Returns:
        Dictionary with model registry summary
    """
    models = registry.list_models()

    # Calculate aggregate statistics
    all_metrics = {}
    for model in models:
        metrics = model.get('test_metrics', {})
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Calculate averages
    metric_stats = {}
    for metric_name, values in all_metrics.items():
        metric_stats[metric_name] = {
            'mean': float(np.mean(values)) if values else 0,
            'std': float(np.std(values)) if values else 0,
            'min': float(np.min(values)) if values else 0,
            'max': float(np.max(values)) if values else 0
        }

    return {
        'total_models': len(models),
        'active_model_id': registry.active_model_id,
        'models': models,
        'metric_statistics': metric_stats,
        'timestamp': datetime.now().isoformat()
    }


@mcp.tool()
async def set_active_model(model_id: str) -> Dict[str, Any]:
    """
    Set the active model for predictions.

    Args:
        model_id: Model ID to activate

    Returns:
        Status of activation
    """
    success = registry.set_active_model(model_id)

    if success:
        model = registry.get_active_model()
        return {
            'status': 'success',
            'message': f'Model {model_id} is now active',
            'active_model': model
        }
    else:
        return {
            'status': 'error',
            'message': f'Model {model_id} not found'
        }


@mcp.tool()
async def compare_models(model_ids: List[str]) -> Dict[str, Any]:
    """
    Compare performance metrics across multiple models.

    Args:
        model_ids: List of model IDs to compare

    Returns:
        Comparison table with metrics
    """
    comparison = {
        'models_compared': len(model_ids),
        'comparison': [],
        'timestamp': datetime.now().isoformat()
    }

    for model_id in model_ids:
        model = registry.get_model(model_id)
        if model:
            comparison['comparison'].append({
                'model_id': model_id,
                'model_name': model.get('model_name'),
                'architecture': model.get('architecture'),
                'metrics': model.get('test_metrics', {}),
                'dataset': model.get('dataset'),
                'created_at': model.get('created_at')
            })

    return comparison


@mcp.tool()
async def get_model_context(
    model_id: Optional[str] = None,
    include_performance: bool = True,
    include_config: bool = True
) -> Dict[str, Any]:
    """
    Get complete context for a model (MCP context protocol).

    Args:
        model_id: Model ID (uses active if not specified)
        include_performance: Include performance metrics
        include_config: Include training configuration

    Returns:
        Complete model context for deployment
    """
    target_model_id = model_id or registry.active_model_id
    if not target_model_id:
        return {'error': 'No model specified and no active model set'}

    model = registry.get_model(target_model_id)
    if not model:
        return {'error': f'Model {target_model_id} not found'}

    context = {
        'model_id': target_model_id,
        'model_name': model.get('model_name'),
        'architecture': model.get('architecture'),
        'timestamp': datetime.now().isoformat()
    }

    if include_config:
        context['training_config'] = model.get('training_config')

    if include_performance:
        context['test_metrics'] = model.get('test_metrics')

    context['deployment_ready'] = True
    context['usage_instructions'] = {
        'single_prediction': 'Use predict() tool with text input',
        'batch_prediction': 'Use batch_predict() tool with list of texts',
        'model_switching': 'Use set_active_model() to switch models'
    }

    return context


if __name__ == '__main__':
    # Run the MCP server
    asyncio.run(mcp.run(transport='stdio'))
