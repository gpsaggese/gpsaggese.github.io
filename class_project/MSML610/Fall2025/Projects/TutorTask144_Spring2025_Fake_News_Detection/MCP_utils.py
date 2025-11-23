"""
Utility functions for MCP (Model Context Protocol) operations.

Provides helper functions for:
- Model registry management
- Model versioning and tracking
- Context generation for deployment
- Metrics computation and comparison
- Integration with BERT models
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_utils")


@dataclass
class ModelVersion:
    """Represents a model version in the registry."""
    model_id: str
    model_name: str
    architecture: str
    version: str
    created_at: str
    training_config: Dict[str, Any]
    test_metrics: Dict[str, float]
    dataset: str
    model_path: str
    status: str = 'active'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MCPRegistry:
    """Manages model registry for version control and tracking."""

    def __init__(self, registry_file: str = "deep_learning_registry.json"):
        """
        Initialize registry.

        Args:
            registry_file: Path to registry JSON file
        """
        self.registry_file = Path(registry_file)
        self.models: Dict[str, ModelVersion] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.get('models', {}).items():
                        self.models[model_id] = ModelVersion(**model_data)
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {str(e)}")
                self.models = {}

    def save(self) -> None:
        """Save registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                'models': {
                    model_id: model.to_dict()
                    for model_id, model in self.models.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.models)} models to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {str(e)}")

    def add_model(
        self,
        model_name: str,
        architecture: str,
        training_config: Dict[str, Any],
        test_metrics: Dict[str, float],
        dataset: str,
        model_path: str,
        version: Optional[str] = None
    ) -> str:
        """
        Add a new model to the registry.

        Args:
            model_name: Descriptive model name
            architecture: Architecture description
            training_config: Training configuration
            test_metrics: Performance metrics
            dataset: Dataset used
            model_path: Path to model weights
            version: Optional version string (auto-generated if not provided)

        Returns:
            Generated model ID
        """
        model_id = str(uuid.uuid4())[:8]
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')

        model = ModelVersion(
            model_id=model_id,
            model_name=model_name,
            architecture=architecture,
            version=version,
            created_at=datetime.now().isoformat(),
            training_config=training_config,
            test_metrics=test_metrics,
            dataset=dataset,
            model_path=model_path,
            status='active'
        )

        self.models[model_id] = model
        self.save()
        logger.info(f"Added model {model_id}: {model_name}")
        return model_id

    def get_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID."""
        return self.models.get(model_id)

    def list_models(self) -> List[ModelVersion]:
        """List all models."""
        return list(self.models.values())

    def find_by_name(self, name: str) -> List[ModelVersion]:
        """Find models by name pattern."""
        return [m for m in self.models.values() if name.lower() in m.model_name.lower()]

    def find_by_dataset(self, dataset: str) -> List[ModelVersion]:
        """Find models trained on specific dataset."""
        return [m for m in self.models.values() if m.dataset == dataset]

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            self.save()
            logger.info(f"Deleted model {model_id}")
            return True
        return False


class MetricsComparator:
    """Compares metrics across multiple models."""

    @staticmethod
    def compare_models(
        models: List[ModelVersion],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create comparison DataFrame for models.

        Args:
            models: List of models to compare
            metric_names: Specific metrics to include (all if None)

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if not models:
            return pd.DataFrame()

        data = []
        for model in models:
            row = {
                'model_id': model.model_id,
                'model_name': model.model_name,
                'architecture': model.architecture,
                'dataset': model.dataset,
                'created_at': model.created_at
            }

            # Add metrics
            if metric_names:
                for metric in metric_names:
                    row[metric] = model.test_metrics.get(metric, np.nan)
            else:
                row.update(model.test_metrics)

            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def rank_by_metric(
        models: List[ModelVersion],
        metric: str,
        ascending: bool = False
    ) -> List[Tuple[ModelVersion, float]]:
        """
        Rank models by a specific metric.

        Args:
            models: Models to rank
            metric: Metric name to rank by
            ascending: Sort in ascending order if True

        Returns:
            List of (model, metric_value) tuples sorted by metric
        """
        ranked = [
            (m, m.test_metrics.get(metric, 0))
            for m in models
        ]
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    @staticmethod
    def calculate_statistics(
        models: List[ModelVersion],
        metric: str
    ) -> Dict[str, float]:
        """
        Calculate statistics for a metric across models.

        Args:
            models: Models to analyze
            metric: Metric name

        Returns:
            Statistics (mean, std, min, max, median)
        """
        values = [m.test_metrics.get(metric, np.nan) for m in models]
        values = [v for v in values if not np.isnan(v)]

        if not values:
            return {}

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }


class ContextGenerator:
    """Generates deployment context for models."""

    @staticmethod
    def generate_deployment_context(
        model: ModelVersion,
        include_config: bool = True,
        include_metrics: bool = True,
        include_usage: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete deployment context for a model.

        Args:
            model: Model to generate context for
            include_config: Include training configuration
            include_metrics: Include performance metrics
            include_usage: Include usage instructions

        Returns:
            Complete deployment context
        """
        context = {
            'model_id': model.model_id,
            'model_name': model.model_name,
            'architecture': model.architecture,
            'version': model.version,
            'created_at': model.created_at,
            'dataset': model.dataset,
            'model_path': model.model_path,
            'status': model.status
        }

        if include_config:
            context['training_config'] = model.training_config

        if include_metrics:
            context['test_metrics'] = model.test_metrics

        if include_usage:
            context['deployment_instructions'] = {
                'load_model': f"Load from: {model.model_path}",
                'input_format': 'Text string (max 512 characters)',
                'output_format': 'Binary classification (0=real, 1=fake)',
                'typical_latency': '< 1 second per prediction',
                'batch_support': True,
                'max_batch_size': 32
            }

        context['deployment_ready'] = True
        context['generated_at': datetime.now().isoformat()

        return context

    @staticmethod
    def generate_comparison_context(
        models: List[ModelVersion],
        metric_for_ranking: str = 'f1'
    ) -> Dict[str, Any]:
        """
        Generate context for model comparison.

        Args:
            models: Models to compare
            metric_for_ranking: Metric to rank by

        Returns:
            Comparison context with ranking and statistics
        """
        comparator = MetricsComparator()
        ranked = comparator.rank_by_metric(models, metric_for_ranking)

        comparison = {
            'total_models': len(models),
            'ranking_metric': metric_for_ranking,
            'models_ranked': [
                {
                    'rank': i + 1,
                    'model_id': m.model_id,
                    'model_name': m.model_name,
                    'metric_value': float(score)
                }
                for i, (m, score) in enumerate(ranked)
            ],
            'metric_statistics': comparator.calculate_statistics(models, metric_for_ranking),
            'comparison_table': comparator.compare_models(models).to_dict(),
            'generated_at': datetime.now().isoformat()
        }

        return comparison


class ModelValidator:
    """Validates model versions and configurations."""

    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate training configuration.

        Args:
            config: Training configuration dictionary

        Returns:
            (is_valid, error_message)
        """
        required_fields = ['model_name', 'batch_size', 'learning_rate', 'num_epochs']

        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"

        # Validate value ranges
        if config.get('batch_size', 0) < 1:
            return False, "batch_size must be >= 1"

        if config.get('learning_rate', 0) <= 0:
            return False, "learning_rate must be > 0"

        if config.get('num_epochs', 0) < 1:
            return False, "num_epochs must be >= 1"

        return True, "Valid"

    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate performance metrics.

        Args:
            metrics: Metrics dictionary

        Returns:
            (is_valid, error_message)
        """
        common_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # At least one metric should be present
        if not any(m in metrics for m in common_metrics):
            return False, "No common metrics found"

        # Check value ranges
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                return False, f"{metric} must be numeric"

            if metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if not (0 <= value <= 1):
                    return False, f"{metric} must be between 0 and 1"

        return True, "Valid"


def load_or_create_registry(
    registry_file: str = "deep_learning_registry.json"
) -> MCPRegistry:
    """
    Load existing registry or create a new one.

    Args:
        registry_file: Path to registry file

    Returns:
        MCPRegistry instance
    """
    registry = MCPRegistry(registry_file)
    return registry


def export_registry_to_csv(
    registry: MCPRegistry,
    output_file: str = "model_registry.csv"
) -> None:
    """
    Export registry to CSV format.

    Args:
        registry: MCPRegistry instance
        output_file: Output CSV file path
    """
    models = registry.list_models()
    if not models:
        logger.warning("No models to export")
        return

    # Flatten metrics into separate columns
    rows = []
    for model in models:
        row = {
            'model_id': model.model_id,
            'model_name': model.model_name,
            'architecture': model.architecture,
            'dataset': model.dataset,
            'created_at': model.created_at
        }
        row.update(model.test_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(models)} models to {output_file}")


if __name__ == '__main__':
    # Example usage
    registry = load_or_create_registry()

    # Add example model
    model_id = registry.add_model(
        model_name="BERT Fake News Detector v1.0",
        architecture="DistilBERT-base-uncased",
        training_config={
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 3
        },
        test_metrics={
            'accuracy': 0.6092,
            'precision': 0.6970,
            'recall': 0.5612,
            'f1': 0.6200,
            'roc_auc': 0.55
        },
        dataset='LIAR',
        model_path='models/bert_fake_news_v1'
    )

    print(f"Added model: {model_id}")

    # Generate deployment context
    model = registry.get_model(model_id)
    context = ContextGenerator.generate_deployment_context(model)
    print(json.dumps(context, indent=2))
