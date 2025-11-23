# MCP (Model Context Protocol) API Documentation

## Fake News Detection System

Complete API reference for the MCP server and client for BERT-based fake news detection with model versioning and context-aware deployment.

## Overview

The MCP implementation provides:
- **Model Registry**: Track and manage multiple model versions
- **Predictions**: Single and batch inference with confidence scores
- **Model Versioning**: Complete version history with metadata
- **Context Management**: Deploy models with full context information
- **Model Comparison**: Compare performance across versions

## Architecture

```
┌─────────────┐
│   Client    │  (MCP.client.py)
└──────┬──────┘
       │ stdio protocol
       │
┌──────▼──────┐
│   Server    │  (MCP.server.py)
├─────────────┤
│ - Resources │ (Model metadata)
│ - Tools     │ (Predictions, management)
│ - Registry  │ (Model versioning)
└──────┬──────┘
       │
┌──────▼──────┐
│   Models    │  (BERT, stored locally)
└─────────────┘
```

## Resources (Read-Only)

Resources provide read-only access to model metadata and registry information.

### model://registry

Get the complete model registry with all versions.

**Request:**
```
GET model://registry
```

**Response:**
```json
{
  "total_models": 3,
  "active_model_id": "a1b2c3d4",
  "active_model": {
    "model_id": "a1b2c3d4",
    "model_name": "BERT Fake News v1.0",
    "architecture": "DistilBERT-base-uncased",
    "test_metrics": {
      "accuracy": 0.6092,
      "precision": 0.6970,
      "f1": 0.6200,
      "roc_auc": 0.55
    },
    "dataset": "LIAR",
    "created_at": "2024-11-23T10:30:00"
  },
  "models": [
    {
      "model_id": "a1b2c3d4",
      "model_name": "BERT Fake News v1.0",
      ...
    }
  ],
  "timestamp": "2024-11-23T10:35:15"
}
```

### model://active

Get information about the currently active model.

**Request:**
```
GET model://active
```

**Response:**
```json
{
  "model_id": "a1b2c3d4",
  "model_name": "BERT Fake News v1.0",
  "architecture": "DistilBERT-base-uncased",
  "training_config": {
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 3
  },
  "test_metrics": {
    "accuracy": 0.6092,
    "precision": 0.6970,
    "f1": 0.6200,
    "roc_auc": 0.55
  },
  "dataset": "LIAR",
  "created_at": "2024-11-23T10:30:00",
  "status": "active"
}
```

### model://metrics/{model_id}

Get performance metrics for a specific model.

**Request:**
```
GET model://metrics/a1b2c3d4
```

**Response:**
```json
{
  "model_id": "a1b2c3d4",
  "model_name": "BERT Fake News v1.0",
  "metrics": {
    "accuracy": 0.6092,
    "precision": 0.6970,
    "recall": 0.5612,
    "f1": 0.6200,
    "roc_auc": 0.55
  },
  "dataset": "LIAR",
  "created_at": "2024-11-23T10:30:00"
}
```

### model://architecture/{model_id}

Get architecture details for a specific model.

**Request:**
```
GET model://architecture/a1b2c3d4
```

**Response:**
```json
{
  "model_id": "a1b2c3d4",
  "architecture": "DistilBERT-base-uncased",
  "training_config": {
    "model_name": "distilbert-base-uncased",
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "patience": 1,
    "device": "cuda",
    "use_class_weights": false,
    "max_text_length": 256
  },
  "created_at": "2024-11-23T10:30:00"
}
```

## Tools (Read-Write)

Tools provide methods for making predictions, managing models, and accessing context.

### predict

Make a fake news prediction on input text.

**Parameters:**
- `text` (string, required): News article text to classify (max 512 characters)
- `model_id` (string, optional): Specific model to use (uses active model if not specified)
- `return_confidence` (boolean, optional): Include confidence scores in response (default: true)

**Request:**
```json
{
  "text": "Breaking news: Scientists discover new planet in habitable zone",
  "model_id": "a1b2c3d4",
  "return_confidence": true
}
```

**Response:**
```json
{
  "text_preview": "Breaking news: Scientists discover new planet in ha...",
  "text_length": 62,
  "prediction": 0,
  "label": "Real",
  "model_id": "a1b2c3d4",
  "confidence": {
    "real": 0.8234,
    "fake": 0.1766
  },
  "timestamp": "2024-11-23T10:35:15"
}
```

**Response Fields:**
- `prediction`: 0 = real news, 1 = fake news
- `label`: Human-readable label ("Real" or "Fake")
- `confidence`: Probability scores for each class (only if return_confidence=true)
- `model_id`: ID of the model used for prediction

### batch_predict

Make predictions on multiple text samples.

**Parameters:**
- `texts` (array of strings, required): List of news article texts
- `model_id` (string, optional): Specific model to use (uses active model if not specified)

**Request:**
```json
{
  "texts": [
    "Scientists discover new planet",
    "Famous actor involved in scandal",
    "New government policy announced"
  ],
  "model_id": "a1b2c3d4"
}
```

**Response:**
```json
{
  "total_samples": 3,
  "fake_count": 1,
  "real_count": 2,
  "fake_percentage": 33.33,
  "model_id": "a1b2c3d4",
  "predictions": [
    {
      "text_preview": "Scientists discover new planet",
      "prediction": 0,
      "label": "Real"
    },
    {
      "text_preview": "Famous actor involved in scandal",
      "prediction": 1,
      "label": "Fake"
    },
    {
      "text_preview": "New government policy announced",
      "prediction": 0,
      "label": "Real"
    }
  ],
  "timestamp": "2024-11-23T10:35:15"
}
```

### register_model_version

Register a new model version in the registry.

**Parameters:**
- `model_name` (string, required): Descriptive name for the model
- `architecture` (string, required): Model architecture description
- `training_config` (object, required): Training configuration parameters
- `test_metrics` (object, required): Performance metrics on test set
- `dataset` (string, required): Dataset used for training
- `model_path` (string, required): Path to saved model weights

**Request:**
```json
{
  "model_name": "BERT Fake News v2.0",
  "architecture": "DistilBERT-base-uncased with class weights",
  "training_config": {
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 5,
    "use_class_weights": true
  },
  "test_metrics": {
    "accuracy": 0.6215,
    "precision": 0.7100,
    "recall": 0.5823,
    "f1": 0.6380,
    "roc_auc": 0.58
  },
  "dataset": "LIAR",
  "model_path": "models/bert_fake_news_v2"
}
```

**Response:**
```json
{
  "status": "registered",
  "model_id": "x9y8z7w6",
  "model_name": "BERT Fake News v2.0",
  "created_at": "2024-11-23T10:35:15",
  "architecture": "DistilBERT-base-uncased with class weights",
  "training_config": {...},
  "test_metrics": {...},
  "dataset": "LIAR",
  "model_path": "models/bert_fake_news_v2"
}
```

### list_all_models

List all registered model versions with their metrics.

**Request:**
```json
{}
```

**Response:**
```json
{
  "total_models": 2,
  "active_model_id": "a1b2c3d4",
  "models": [
    {
      "model_id": "a1b2c3d4",
      "model_name": "BERT Fake News v1.0",
      "architecture": "DistilBERT-base-uncased",
      "test_metrics": {...},
      "dataset": "LIAR",
      "created_at": "2024-11-23T10:30:00"
    },
    {
      "model_id": "x9y8z7w6",
      "model_name": "BERT Fake News v2.0",
      "architecture": "DistilBERT-base-uncased with class weights",
      "test_metrics": {...},
      "dataset": "LIAR",
      "created_at": "2024-11-23T10:35:15"
    }
  ],
  "metric_statistics": {
    "accuracy": {
      "mean": 0.6153,
      "std": 0.0061,
      "min": 0.6092,
      "max": 0.6215
    },
    "f1": {
      "mean": 0.6290,
      "std": 0.0090,
      "min": 0.6200,
      "max": 0.6380
    }
  },
  "timestamp": "2024-11-23T10:35:15"
}
```

### set_active_model

Set the active model for predictions.

**Parameters:**
- `model_id` (string, required): Model ID to activate

**Request:**
```json
{
  "model_id": "x9y8z7w6"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model x9y8z7w6 is now active",
  "active_model": {
    "model_id": "x9y8z7w6",
    "model_name": "BERT Fake News v2.0",
    "architecture": "DistilBERT-base-uncased with class weights",
    "test_metrics": {...},
    "dataset": "LIAR",
    "created_at": "2024-11-23T10:35:15"
  }
}
```

### compare_models

Compare performance metrics across multiple models.

**Parameters:**
- `model_ids` (array of strings, required): List of model IDs to compare

**Request:**
```json
{
  "model_ids": ["a1b2c3d4", "x9y8z7w6"]
}
```

**Response:**
```json
{
  "models_compared": 2,
  "comparison": [
    {
      "model_id": "a1b2c3d4",
      "model_name": "BERT Fake News v1.0",
      "architecture": "DistilBERT-base-uncased",
      "metrics": {
        "accuracy": 0.6092,
        "precision": 0.6970,
        "f1": 0.6200,
        "roc_auc": 0.55
      },
      "dataset": "LIAR",
      "created_at": "2024-11-23T10:30:00"
    },
    {
      "model_id": "x9y8z7w6",
      "model_name": "BERT Fake News v2.0",
      "architecture": "DistilBERT-base-uncased with class weights",
      "metrics": {
        "accuracy": 0.6215,
        "precision": 0.7100,
        "f1": 0.6380,
        "roc_auc": 0.58
      },
      "dataset": "LIAR",
      "created_at": "2024-11-23T10:35:15"
    }
  ],
  "timestamp": "2024-11-23T10:35:15"
}
```

### get_model_context

Get complete context for a model (MCP context protocol).

**Parameters:**
- `model_id` (string, optional): Model ID (uses active model if not specified)
- `include_performance` (boolean, optional): Include performance metrics (default: true)
- `include_config` (boolean, optional): Include training configuration (default: true)

**Request:**
```json
{
  "model_id": "a1b2c3d4",
  "include_performance": true,
  "include_config": true
}
```

**Response:**
```json
{
  "model_id": "a1b2c3d4",
  "model_name": "BERT Fake News v1.0",
  "architecture": "DistilBERT-base-uncased",
  "training_config": {
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 3
  },
  "test_metrics": {
    "accuracy": 0.6092,
    "precision": 0.6970,
    "f1": 0.6200,
    "roc_auc": 0.55
  },
  "timestamp": "2024-11-23T10:35:15",
  "deployment_ready": true,
  "usage_instructions": {
    "single_prediction": "Use predict() tool with text input",
    "batch_prediction": "Use batch_predict() tool with list of texts",
    "model_switching": "Use set_active_model() to switch models"
  }
}
```

## Python Client API

### FakeNewsMCPClient

High-level client for interacting with the MCP server.

#### Initialization

```python
from MCP.client import FakeNewsMCPClient

# With context manager (recommended)
async with FakeNewsMCPClient() as client:
    result = await client.predict("Text to classify")

# Manual connection management
client = FakeNewsMCPClient()
await client.connect()
result = await client.predict("Text to classify")
await client.disconnect()
```

#### Methods

```python
# Make predictions
result = await client.predict(text, model_id=None, return_confidence=True)
result = await client.batch_predict(texts, model_id=None)

# Model management
model_id = await client.register_model(
    model_name="...",
    architecture="...",
    training_config={...},
    test_metrics={...},
    dataset="...",
    model_path="..."
)
await client.set_active_model(model_id)

# Model information
models = await client.list_models()
registry = await client.get_registry()
active = await client.get_active_model()
metrics = await client.get_model_metrics(model_id)
architecture = await client.get_model_architecture(model_id)

# Model comparison and context
comparison = await client.compare_models(model_ids)
context = await client.get_model_context(model_id)
```

## Error Handling

All API responses include error information when applicable:

```json
{
  "error": "Model x9y8z7w6 not found",
  "status": "error"
}
```

Common error codes:
- `"No model specified and no active model set"` - Must specify model_id or set active model
- `"Model {id} not found"` - Model ID doesn't exist in registry
- `"Model {id} not found or failed to load"` - Model exists but couldn't be loaded
- `"Prediction failed: {reason}"` - Error during prediction inference

## Examples

### Python Usage

```python
import asyncio
from MCP.client import FakeNewsMCPClient

async def main():
    async with FakeNewsMCPClient() as client:
        # Make a prediction
        result = await client.predict(
            "Breaking news: Scientists discover new planet"
        )
        print(f"Prediction: {result['label']} ({result['confidence']})")

        # Batch predictions
        batch_result = await client.batch_predict([
            "Article 1",
            "Article 2",
            "Article 3"
        ])
        print(f"Fake percentage: {batch_result['fake_percentage']}%")

        # Get registry
        registry = await client.get_registry()
        print(f"Total models: {registry['total_models']}")

        # Compare models
        comparison = await client.compare_models(["model1", "model2"])
        print(comparison)

asyncio.run(main())
```

## See Also

- [MCP.example.ipynb](MCP.example.ipynb) - Interactive tutorial with examples
- [MCP_utils.py](MCP_utils.py) - Utility functions for model management
- [BERT.API.md](BERT.API.md) - BERT model API documentation
