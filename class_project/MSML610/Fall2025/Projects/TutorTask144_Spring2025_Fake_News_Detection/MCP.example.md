# MCP (Model Context Protocol) - Fake News Detection Example

Complete end-to-end guide for using the MCP system for fake news detection with model versioning, comparison, and deployment.

## Overview

This guide demonstrates:
1. **Starting the MCP server**
2. **Connecting with the client**
3. **Making predictions**
4. **Managing model versions**
5. **Comparing models**
6. **Deploying with context**

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify BERT model files are in place
ls models/
```

## Part 1: Server Setup

### Starting the MCP Server

The MCP server provides a Model Context Protocol interface to the BERT fake news detection model.

```bash
# Method 1: Direct stdio execution (for testing)
python MCP.server.py

# Method 2: Docker container (for deployment)
docker build -t fake-news-mcp:latest .
docker run -p 8888:8888 fake-news-mcp:latest
```

The server:
- Manages model registry (versions, metrics, configuration)
- Provides resources for metadata access
- Implements tools for predictions and model management
- Maintains model cache for fast inference

### Server Output

```
2024-11-23 10:30:00 - fake_news_mcp - INFO - Model Registry initialized
2024-11-23 10:30:01 - fake_news_mcp - INFO - Registered model: a1b2c3d4
```

## Part 2: Client Usage

### Basic Client Setup

```python
import asyncio
from MCP.client import FakeNewsMCPClient

# Initialize async client
async def main():
    async with FakeNewsMCPClient() as client:
        # Client is now connected
        # Use client methods here
        pass

# Run async client
asyncio.run(main())
```

### Getting Registry Information

```python
async def view_registry():
    """View all registered models."""
    async with FakeNewsMCPClient() as client:
        # Get complete registry
        registry = await client.get_registry()

        print(f"Total models: {registry['total_models']}")
        print(f"Active model: {registry['active_model_id']}")

        # Print all models
        for model in registry['models']:
            print(f"\nModel: {model['model_id']}")
            print(f"  Name: {model['model_name']}")
            print(f"  Architecture: {model['architecture']}")
            print(f"  Accuracy: {model['test_metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {model['test_metrics']['f1']:.4f}")

asyncio.run(view_registry())
```

**Output:**
```
Total models: 1
Active model: a1b2c3d4

Model: a1b2c3d4
  Name: BERT Fake News v1.0
  Architecture: DistilBERT-base-uncased
  Accuracy: 0.6092
  F1-Score: 0.6200
```

## Part 3: Making Predictions

### Single Text Prediction

```python
async def predict_single():
    """Make a single prediction with confidence scores."""
    async with FakeNewsMCPClient() as client:
        text = "Breaking news: Scientists discover new planet in habitable zone"

        result = await client.predict(
            text=text,
            return_confidence=True
        )

        print(f"Text: {result['text_preview']}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence:")
        print(f"  Real: {result['confidence']['real']:.2%}")
        print(f"  Fake: {result['confidence']['fake']:.2%}")

asyncio.run(predict_single())
```

**Output:**
```
Text: Breaking news: Scientists discover new planet in ha...
Prediction: Real
Confidence:
  Real: 82.34%
  Fake: 17.66%
```

### Batch Predictions

```python
async def predict_batch():
    """Make predictions on multiple texts."""
    async with FakeNewsMCPClient() as client:
        texts = [
            "Scientists discover new planet in habitable zone",
            "Politician involved in major scandal with no evidence",
            "New government policy announced to reduce carbon emissions",
            "Celebrity endorses miracle cure that cures all diseases",
            "Study shows increased exercise improves health outcomes"
        ]

        result = await client.batch_predict(texts)

        print(f"Total articles: {result['total_samples']}")
        print(f"Real articles: {result['real_count']}")
        print(f"Fake articles: {result['fake_count']}")
        print(f"Fake percentage: {result['fake_percentage']:.1f}%")

        print("\nDetailed predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['text_preview']}")
            print(f"   → {pred['label']}")

asyncio.run(predict_batch())
```

**Output:**
```
Total articles: 5
Real articles: 3
Fake articles: 2
Fake percentage: 40.0%

Detailed predictions:
1. Scientists discover new planet in habitable zone
   → Real
2. Politician involved in major scandal with no evidence
   → Fake
3. New government policy announced to reduce carbon emiss...
   → Real
4. Celebrity endorses miracle cure that cures all diseases
   → Fake
5. Study shows increased exercise improves health outcomes
   → Real
```

## Part 4: Model Management

### Registering a New Model

```python
async def register_new_model():
    """Register a new model version in the registry."""
    async with FakeNewsMCPClient() as client:
        # Register new model with improved metrics
        model_id = await client.register_model(
            model_name="BERT Fake News v2.0",
            architecture="DistilBERT-base-uncased with class weights",
            training_config={
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 5,
                'use_class_weights': True
            },
            test_metrics={
                'accuracy': 0.6215,
                'precision': 0.7100,
                'recall': 0.5823,
                'f1': 0.6380,
                'roc_auc': 0.58
            },
            dataset='LIAR',
            model_path='models/bert_fake_news_v2'
        )

        print(f"Registered model: {model_id}")
        print(f"Status: {model_id['status']}")
        print(f"Model name: {model_id['model_name']}")

asyncio.run(register_new_model())
```

**Output:**
```
Registered model: x9y8z7w6
Status: registered
Model name: BERT Fake News v2.0
```

### Switching Active Model

```python
async def switch_model():
    """Switch to a different model for predictions."""
    async with FakeNewsMCPClient() as client:
        # Get list of models
        models = await client.list_models()
        model_ids = [m['model_id'] for m in models['models']]

        # Switch to second model
        if len(model_ids) > 1:
            result = await client.set_active_model(model_ids[1])

            print(f"Switched to: {result['active_model']['model_name']}")
            print(f"New active model ID: {result['active_model']['model_id']}")
        else:
            print("Only one model available")

asyncio.run(switch_model())
```

**Output:**
```
Switched to: BERT Fake News v2.0
New active model ID: x9y8z7w6
```

## Part 5: Model Comparison

### Comparing Model Performance

```python
async def compare_models():
    """Compare performance metrics across models."""
    async with FakeNewsMCPClient() as client:
        # Get all models
        models = await client.list_models()
        model_ids = [m['model_id'] for m in models['models']]

        # Compare models
        if len(model_ids) > 1:
            comparison = await client.compare_models(model_ids)

            print(f"Comparing {comparison['models_compared']} models:")
            print()

            for model_comp in comparison['comparison']:
                print(f"Model: {model_comp['model_name']}")
                print(f"  ID: {model_comp['model_id']}")
                print(f"  Architecture: {model_comp['architecture']}")
                print(f"  Metrics:")
                for metric, value in model_comp['metrics'].items():
                    print(f"    {metric}: {value:.4f}")
                print()
        else:
            print("Need at least 2 models to compare")

asyncio.run(compare_models())
```

**Output:**
```
Comparing 2 models:

Model: BERT Fake News v1.0
  ID: a1b2c3d4
  Architecture: DistilBERT-base-uncased
  Metrics:
    accuracy: 0.6092
    precision: 0.6970
    f1: 0.6200
    roc_auc: 0.5500

Model: BERT Fake News v2.0
  ID: x9y8z7w6
  Architecture: DistilBERT-base-uncased with class weights
  Metrics:
    accuracy: 0.6215
    precision: 0.7100
    f1: 0.6380
    roc_auc: 0.5800
```

## Part 6: Deployment Context

### Getting Model Context

```python
async def get_deployment_context():
    """Get complete context for model deployment."""
    async with FakeNewsMCPClient() as client:
        # Get context for active model
        context = await client.get_model_context()

        print(f"Model: {context['model_name']}")
        print(f"Architecture: {context['architecture']}")
        print(f"Deployment ready: {context['deployment_ready']}")

        print(f"\nTraining configuration:")
        for key, value in context['training_config'].items():
            print(f"  {key}: {value}")

        print(f"\nPerformance metrics:")
        for metric, value in context['test_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nUsage instructions:")
        for instruction, desc in context['usage_instructions'].items():
            print(f"  {instruction}: {desc}")

asyncio.run(get_deployment_context())
```

**Output:**
```
Model: BERT Fake News v1.0
Architecture: DistilBERT-base-uncased
Deployment ready: True

Training configuration:
  model_name: distilbert-base-uncased
  batch_size: 16
  learning_rate: 2e-05
  num_epochs: 3
  warmup_ratio: 0.1
  max_grad_norm: 1.0

Performance metrics:
  accuracy: 0.6092
  precision: 0.6970
  f1: 0.6200
  roc_auc: 0.5500

Usage instructions:
  single_prediction: Use predict() tool with text input
  batch_prediction: Use batch_predict() tool with list of texts
  model_switching: Use set_active_model() to switch models
```

## Part 7: Advanced Usage

### Using Specific Model for Prediction

```python
async def predict_with_specific_model():
    """Make prediction with a specific model (not active)."""
    async with FakeNewsMCPClient() as client:
        text = "Major news event announced today"

        # Get all models
        models = await client.list_models()

        # Predict with each model
        for model in models['models']:
            result = await client.predict(
                text=text,
                model_id=model['model_id'],
                return_confidence=True
            )

            print(f"Model: {model['model_name']}")
            print(f"  Prediction: {result['label']}")
            print(f"  Confidence: {result['confidence']}")
            print()

asyncio.run(predict_with_specific_model())
```

### Getting Detailed Metrics

```python
async def get_model_metrics():
    """Get detailed metrics for a specific model."""
    async with FakeNewsMCPClient() as client:
        # Get active model
        active = await client.get_active_model()

        # Get metrics
        metrics = await client.get_model_metrics(active['model_id'])

        print(f"Model: {metrics['model_name']}")
        print(f"Dataset: {metrics['dataset']}")
        print(f"\nMetrics:")
        for metric, value in metrics['metrics'].items():
            print(f"  {metric}: {value:.4f}")

asyncio.run(get_model_metrics())
```

## Part 8: Error Handling

### Handling API Errors

```python
async def handle_errors():
    """Demonstrate error handling."""
    async with FakeNewsMCPClient() as client:
        # Try to predict without active model (will fail)
        try:
            result = await client.predict(
                text="Some text",
                model_id="nonexistent_id"
            )

            if 'error' in result:
                print(f"Error: {result['error']}")
                print(f"Status: {result['status']}")
        except Exception as e:
            print(f"Exception: {str(e)}")

asyncio.run(handle_errors())
```

**Output:**
```
Error: Model nonexistent_id not found or failed to load
Status: error
```

## Part 9: Complete End-to-End Example

```python
async def end_to_end_example():
    """Complete end-to-end example."""
    async with FakeNewsMCPClient() as client:
        print("=" * 60)
        print("MCP FAKE NEWS DETECTION - END-TO-END EXAMPLE")
        print("=" * 60)

        # 1. View registry
        print("\n1. Viewing model registry...")
        registry = await client.get_registry()
        print(f"   Total models: {registry['total_models']}")

        # 2. Make prediction
        print("\n2. Making prediction...")
        result = await client.predict(
            "Scientists discover breakthrough in renewable energy"
        )
        print(f"   Prediction: {result['label']}")
        print(f"   Confidence: {result['confidence']}")

        # 3. Batch predict
        print("\n3. Batch prediction...")
        batch = await client.batch_predict([
            "Text 1",
            "Text 2",
            "Text 3"
        ])
        print(f"   Fake articles: {batch['fake_count']}/{batch['total_samples']}")

        # 4. Get active model info
        print("\n4. Active model info...")
        active = await client.get_active_model()
        print(f"   Model: {active['model_name']}")
        print(f"   Accuracy: {active['test_metrics']['accuracy']:.4f}")

        # 5. Get deployment context
        print("\n5. Getting deployment context...")
        context = await client.get_model_context()
        print(f"   Ready: {context['deployment_ready']}")

        print("\n" + "=" * 60)

asyncio.run(end_to_end_example())
```

## Part 10: Utilities

### Using MCP_utils for Registry Management

```python
from MCP_utils import load_or_create_registry, MetricsComparator, ContextGenerator

# Load or create registry
registry = load_or_create_registry()

# List all models
all_models = registry.list_models()
print(f"Models in registry: {len(all_models)}")

# Compare metrics
comparator = MetricsComparator()
comparison_df = comparator.compare_models(all_models, ['accuracy', 'f1', 'roc_auc'])
print(comparison_df)

# Rank models by F1
ranked = comparator.rank_by_metric(all_models, 'f1')
for model, f1_score in ranked:
    print(f"{model.model_name}: {f1_score:.4f}")

# Generate deployment context
context = ContextGenerator.generate_deployment_context(all_models[0])
print(f"Deployment context generated for: {context['model_name']}")
```

## Part 11: Docker Deployment

### Building and Running with Docker

```bash
# Build image
docker build -t fake-news-mcp:latest .

# Run server in container
docker run -d \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name fake-news-server \
  fake-news-mcp:latest

# Connect client to containerized server
# Update MCP.client.py to connect to container's socket

# View logs
docker logs fake-news-server

# Stop container
docker stop fake-news-server
```

## Summary

The MCP system provides:

✓ **Model Registry** - Track multiple versions with metadata
✓ **Predictions** - Single and batch inference
✓ **Versioning** - Complete model history
✓ **Context** - Deployment-ready context for each model
✓ **Comparison** - Analyze performance across versions
✓ **Management** - Register, activate, and switch models
✓ **Deployment** - Docker and async support for production use

## See Also

- [MCP.API.md](MCP.API.md) - Complete API reference
- [BERT.API.md](BERT.API.md) - BERT model API
- [MCP_utils.py](MCP_utils.py) - Utility functions
- [README.md](README.md) - Project documentation
