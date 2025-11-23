# MCP (Model Context Protocol) Architecture

## Project Focus: MCP-Driven Fake News Detection System

This project demonstrates **Model Context Protocol (MCP)** as the primary architecture for managing, versioning, and deploying fake news detection models.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP-BASED SYSTEM                          │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   MCP Client        │  ← Application/User Interface
│ (MCP.client.py)     │
└──────────┬──────────┘
           │
           │ stdio protocol (Model Context Protocol)
           │
┌──────────▼──────────────────────────────────────────────┐
│                  MCP SERVER INSTANCE                     │
│              (MCP.server.py)                            │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │     MCP RESOURCES (Read-Only)                   │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ • model://registry      → Model registry JSON   │    │
│  │ • model://active        → Active model info     │    │
│  │ • model://metrics/{id}  → Performance metrics   │    │
│  │ • model://architecture/ → Architecture details  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │     MCP TOOLS (Read-Write)                      │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ • predict()              → Single prediction    │    │
│  │ • batch_predict()        → Batch predictions    │    │
│  │ • register_model_version → Add model version    │    │
│  │ • list_all_models()      → List all versions    │    │
│  │ • set_active_model()     → Switch active model  │    │
│  │ • compare_models()       → Compare performance  │    │
│  │ • get_model_context()    → Deployment context  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │     MODEL REGISTRY                              │    │
│  │ (deep_learning_registry.json)                   │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ • Version history                               │    │
│  │ • Model metadata                                │    │
│  │ • Performance metrics                           │    │
│  │ • Training configuration                        │    │
│  │ • Model paths                                   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │     MODEL CACHE & LOADER                        │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ • Load models on-demand                         │    │
│  │ • Cache for fast inference                      │    │
│  │ • Version switching                             │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
           │
           │ Model inference
           │
┌──────────▼──────────────────────────┐
│   BERT/LSTM Models                   │
│   (Loaded from disk/cache)           │
└──────────────────────────────────────┘
```

---

## MCP Resources

### 1. **model://registry**
Complete model registry with all versions.

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
      "f1": 0.6200
    }
  },
  "models": [
    { "model_id": "a1b2c3d4", ... },
    { "model_id": "x9y8z7w6", ... }
  ]
}
```

### 2. **model://active**
Information about currently active model.

```json
{
  "model_id": "a1b2c3d4",
  "model_name": "BERT Fake News v1.0",
  "architecture": "DistilBERT-base-uncased",
  "status": "active",
  "training_config": {
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 3
  },
  "test_metrics": {
    "accuracy": 0.6092,
    "precision": 0.6970,
    "f1": 0.6200
  }
}
```

### 3. **model://metrics/{model_id}**
Performance metrics for specific model.

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

### 4. **model://architecture/{model_id}**
Architecture and configuration details.

```json
{
  "model_id": "a1b2c3d4",
  "architecture": "DistilBERT-base-uncased",
  "training_config": {
    "model_name": "distilbert-base-uncased",
    "batch_size": 16,
    "learning_rate": 0.00002,
    "num_epochs": 3,
    "warmup_ratio": 0.1
  }
}
```

---

## MCP Tools

### 1. **predict(text, model_id, return_confidence)**
Make single prediction.

```python
result = await client.predict(
    text="Breaking news article text",
    model_id="a1b2c3d4",
    return_confidence=True
)

# Returns:
{
    "prediction": 0,
    "label": "Real",
    "confidence": {
        "real": 0.82,
        "fake": 0.18
    },
    "model_id": "a1b2c3d4"
}
```

### 2. **batch_predict(texts, model_id)**
Make predictions on multiple texts.

```python
results = await client.batch_predict(
    texts=["text1", "text2", "text3"],
    model_id="a1b2c3d4"
)

# Returns:
{
    "total_samples": 3,
    "fake_count": 1,
    "real_count": 2,
    "fake_percentage": 33.33,
    "predictions": [...]
}
```

### 3. **register_model_version(...)**
Register new model version.

```python
model_id = await client.register_model(
    model_name="BERT Fake News v2.0",
    architecture="DistilBERT-base-uncased with class weights",
    training_config={
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 5
    },
    test_metrics={
        "accuracy": 0.6215,
        "precision": 0.7100,
        "f1": 0.6380
    },
    dataset="LIAR",
    model_path="models/bert_v2"
)
```

### 4. **list_all_models()**
List all registered models.

```python
models = await client.list_all_models()

# Returns:
{
    "total_models": 2,
    "models": [
        {
            "model_id": "a1b2c3d4",
            "model_name": "BERT v1.0",
            "accuracy": 0.6092
        },
        {
            "model_id": "x9y8z7w6",
            "model_name": "BERT v2.0",
            "accuracy": 0.6215
        }
    ]
}
```

### 5. **set_active_model(model_id)**
Switch active model for predictions.

```python
result = await client.set_active_model("x9y8z7w6")

# Returns:
{
    "status": "success",
    "message": "Model x9y8z7w6 is now active",
    "active_model": {...}
}
```

### 6. **compare_models(model_ids)**
Compare performance across models.

```python
comparison = await client.compare_models(
    ["a1b2c3d4", "x9y8z7w6"]
)

# Returns detailed comparison table
```

### 7. **get_model_context(model_id)**
Get full context for deployment.

```python
context = await client.get_model_context(
    model_id="a1b2c3d4",
    include_performance=True,
    include_config=True
)

# Returns complete deployment context
```

---

## Model Context Protocol Flow

### Registration Flow
```
1. Train new model locally
   ↓
2. Call register_model_version()
   ↓
3. MCP Server stores in registry
   ↓
4. Returns unique model_id
   ↓
5. Registry persisted to JSON
```

### Prediction Flow
```
1. Client calls predict(text)
   ↓
2. MCP Server receives request
   ↓
3. Load active model from cache
   ↓
4. Tokenize and preprocess text
   ↓
5. Run inference
   ↓
6. Return prediction with metadata
```

### Model Switching Flow
```
1. Client calls set_active_model(model_id)
   ↓
2. MCP Server validates model exists
   ↓
3. Updates active model in registry
   ↓
4. Subsequent predictions use new model
   ↓
5. Seamless switching without restart
```

---

## Model Registry Structure

```
deep_learning_registry.json
├── models
│   ├── a1b2c3d4
│   │   ├── model_id: "a1b2c3d4"
│   │   ├── model_name: "BERT Fake News v1.0"
│   │   ├── architecture: "DistilBERT-base-uncased"
│   │   ├── training_config: {...}
│   │   ├── test_metrics: {...}
│   │   ├── dataset: "LIAR"
│   │   ├── model_path: "models/bert_v1"
│   │   ├── created_at: "2024-11-23T10:30:00"
│   │   └── status: "active"
│   │
│   └── x9y8z7w6
│       ├── model_id: "x9y8z7w6"
│       ├── model_name: "BERT Fake News v2.0"
│       └── ...
│
├── active_model_id: "a1b2c3d4"
└── last_updated: "2024-11-23T15:45:00"
```

---

## MCP Implementation Components

### 1. **MCP Server** (MCP.server.py)
- FastMCP framework integration
- ModelRegistry for persistence
- Resource endpoints (read-only)
- Tool implementations (read-write)
- Model cache management
- Error handling

### 2. **MCP Client** (MCP.client.py)
- Async Python interface
- Context manager support
- High-level API methods
- Connection management
- Error handling

### 3. **MCP Utilities** (MCP_utils.py)
- MCPRegistry: Load/save operations
- MetricsComparator: Model comparison
- ContextGenerator: Deployment context
- ModelValidator: Configuration validation

---

## Key MCP Benefits

### 1. **Version Management**
- Track multiple model versions
- Easy rollback to previous versions
- Complete history with metrics

### 2. **Context-Aware Deployment**
- Models deployed with full context
- Metadata accessible via resources
- Configuration stored with model

### 3. **Model Switching**
- Switch models without restart
- No downtime between versions
- Seamless routing

### 4. **Metrics Tracking**
- Performance metrics per version
- Historical comparison
- Optimization tracking

### 5. **Scalability**
- Supports unlimited model versions
- Cache for fast access
- Async operations

---

## Usage Examples

### Start MCP Server
```bash
python MCP.server.py
```

### Python Client Usage
```python
import asyncio
from MCP.client import FakeNewsMCPClient

async def main():
    async with FakeNewsMCPClient() as client:
        # Get registry
        registry = await client.get_registry()

        # Make prediction
        result = await client.predict("Article text")

        # Register new model
        model_id = await client.register_model(...)

        # Switch active model
        await client.set_active_model(model_id)

asyncio.run(main())
```

---

## MCP Protocol Specification

### Transport
- **Protocol**: stdio (standard input/output)
- **Format**: JSON-RPC 2.0
- **Encoding**: UTF-8

### Request Format
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "resources/read",
    "params": {
        "uri": "model://registry"
    }
}
```

### Response Format
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "contents": [
            {
                "uri": "model://registry",
                "mimeType": "application/json",
                "text": "{...}"
            }
        ]
    }
}
```

---

## Deployment Architecture

### Local Deployment
```
Client → MCP Server (stdio) → Models
```

### Docker Deployment
```
Container (Client + Server + Models) → Exposed via ports
```

### Distributed Deployment
```
Multiple MCP Servers → Load Balancer → Shared Registry
```

---

## Future Extensions

1. **Multi-Model Ensemble**
   - Route to multiple models
   - Aggregate predictions
   - Confidence-based selection

2. **Category-Based Routing**
   - Use category detection
   - Route to category-specific models
   - Improved accuracy per category

3. **Active Learning Integration**
   - Track uncertain predictions
   - Request annotations
   - Retrain models

4. **Monitoring & Observability**
   - Track prediction latency
   - Monitor model drift
   - Performance alerts

---

## References

- **MCP Specification**: https://modelcontextprotocol.io
- **FastMCP Framework**: https://github.com/jlowin/FastMCP
- **Python MCP SDK**: https://github.com/anthropics/python-sdk
