# MCP-Driven Fake News Detection System

**Project Focus: Model Context Protocol (MCP) Implementation**

An MCP-based fake news detection system that demonstrates Model Context Protocol for model management, versioning, and context-aware deployment. Uses DistilBERT fine-tuned on multiple datasets (LIAR, ISOT, FakeNewsNet) as the core ML component.

## Core Features: MCP Implementation

### MCP Server & Client
- **MCP Server** (MCP.server.py): FastMCP implementation with resources and tools
- **MCP Client** (MCP.client.py): Async Python client for MCP server communication
- **Model Registry** (deep_learning_registry.json): Persistent model versioning
- **stdio Protocol**: Model Context Protocol transport layer

### MCP Resources (Read-Only)
- `model://registry` - Complete model registry with all versions
- `model://active` - Active model metadata
- `model://metrics/{id}` - Performance metrics per model
- `model://architecture/{id}` - Architecture and configuration details

### MCP Tools (Read-Write)
- `predict()` - Single text prediction with confidence
- `batch_predict()` - Multi-text batch predictions
- `register_model_version()` - Add new model to registry
- `list_all_models()` - List all registered models
- `set_active_model()` - Switch active model for predictions
- `compare_models()` - Compare performance across models
- `get_model_context()` - Get full deployment context

### Supporting Features
- DistilBERT fine-tuning for text classification
- Multi-dataset support (LIAR, ISOT, FakeNewsNet)
- Lazy tokenization for memory efficiency
- Context-aware model deployment
- Docker containerization support

### Project Structure

```
TutorTask144_Spring2025_Fake_News_Detection/
├── BERT Model Implementation
│   ├── bert_utils.py                 # Core BERT utilities (550+ lines)
│   │   ├── DataConfig                # Data loading configuration
│   │   ├── TrainingConfig            # Training hyperparameters
│   │   ├── ModelMetrics              # Evaluation metrics container
│   │   ├── BertTextDataset           # PyTorch dataset with lazy tokenization
│   │   ├── BertModelWrapper          # Main model training/inference wrapper
│   │   └── DataLoader                # Multi-dataset loader utility
│   ├── train_bert_liar_only.py       # Optimized BERT trainer for LIAR (438 lines)
│   ├── train_bert_model.py           # Multi-dataset BERT trainer (390 lines)
│   ├── eval_bert_liar.py             # Comprehensive evaluation script (212 lines)
│   ├── BERT.API.md                   # Complete API documentation
│   ├── BERT.API.ipynb                # Interactive API tutorial notebook
│   ├── BERT.example.md               # Full implementation guide with examples
│   ├── BERT.example.ipynb            # End-to-end example notebook
│   └── BERT_IMPLEMENTATION_REPORT.md # Detailed implementation analysis
│
├── MCP (Model Context Protocol) System
│   ├── MCP.server.py                 # FastMCP server with resources & tools
│   │   ├── ModelRegistry             # Version & metadata management
│   │   ├── Resources                 # model://registry, model://active
│   │   └── Tools                     # predict, batch_predict, compare_models
│   ├── MCP.client.py                 # Async MCP client for Python
│   │   └── FakeNewsMCPClient         # High-level client interface
│   ├── MCP_utils.py                  # Utilities for registry management
│   │   ├── MCPRegistry               # Load/save model versions
│   │   ├── MetricsComparator         # Model performance comparison
│   │   ├── ContextGenerator          # Deployment context generation
│   │   └── ModelValidator            # Config & metrics validation
│   ├── MCP.API.md                    # Complete API reference
│   ├── MCP.example.md                # End-to-end usage guide
│   ├── deep_learning_registry.json   # Model registry (auto-generated)
│   └── MCP_server/                   # Optional Docker-optimized server
│
├── Configuration & Deployment
│   ├── requirements.txt              # Python dependencies (includes MCP)
│   ├── Dockerfile                    # Docker container setup
│   ├── README.md                     # This file
│   ├── PROJECT_SUMMARY.md            # Project overview
│   ├── COMPLETION_STATUS.md          # Implementation status
│   ├── SUBMISSION_COMPLIANCE.md      # Submission requirements
│   └── bert_eval_results_liar.json   # Evaluation results
│
└── Data Directories
    ├── data/
    │   ├── LIAR/                     # Political fact-checking statements
    │   ├── ISOT/                     # News articles (Reuters, etc.)
    │   └── FakeNewsNet/              # Curated fact-check data
    └── models/                       # Saved model weights
```

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch and transformers libraries
- Docker (optional, for containerization)
- Kaggle datasets (LIAR, ISOT, FakeNewsNet)
- GPU recommended but CPU-compatible

### Quick Start (Local Installation)

1. **Navigate to project directory**
   ```bash
   cd class_project/MSML610/Fall2025/Projects/TutorTask144_Spring2025_Fake_News_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets** (automatic via scripts)
   ```bash
   # Datasets are downloaded on first run of training scripts
   python train_bert_liar_only.py
   ```

4. **Run Jupyter notebooks**
   ```bash
   jupyter notebook BERT.API.ipynb
   ```

### Quick Start with Docker

1. **Build the Docker image**
   ```bash
   docker build -t bert-fake-news:latest .
   ```

2. **Run Jupyter server in container**
   ```bash
   docker run -p 8888:8888 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
     bert-fake-news:latest
   ```

3. **Access Jupyter**
   - Open [http://localhost:8888](http://localhost:8888)
   - Notebooks are available immediately

### Training the Model

**Quick training on LIAR dataset:**
```bash
python train_bert_liar_only.py
```

**Training on multiple datasets:**
```bash
python train_bert_model.py
```

**Evaluate trained model:**
```bash
python eval_bert_liar.py
```

## Documentation

The main documentation files are:

- **BERT.API.md** - API reference with all classes and methods
- **BERT.API.ipynb** - Interactive tutorial showing how to use the API
- **BERT.example.md** - Step-by-step guide on how the project works
- **BERT.example.ipynb** - Working example notebook showing the full workflow

## Datasets

The project uses three datasets:

- **LIAR**: 12,791 political claims from PolitiFact (40.1% fake, 59.9% real)
- **ISOT**: 44,898 news articles from Reuters, Bloomberg, CNN, BBC (52.3% fake, 47.7% real)
- **FakeNewsNet**: 422 curated fact-checks from PolitiFact and BuzzFeed (balanced)

## Model

Uses DistilBERT (lightweight transformer with 6 layers, 12 attention heads, 66.4M parameters). The implementation includes:

- Lazy tokenization for memory efficiency (200MB vs 2GB with eager tokenization)
- AdamW optimizer with 10% warmup and gradient clipping
- Early stopping based on validation loss
- Configurable training parameters (batch size, learning rate, epochs)

## Results

Test accuracy on LIAR dataset: 60.92% (accuracy), 69.70% (precision), 47.60% (F1-score), 0.55 (ROC-AUC). The model correctly identifies 99.56% of real news but only 3.25% of fake news, showing class imbalance issues common with the LIAR dataset. Performance improves with class-weighted loss and more training epochs.

## Model Registry

Trained models are tracked in `deep_learning_registry.json` with metadata including architecture, training config, and test results.

## MCP (Model Context Protocol) System

The project includes a complete MCP implementation for model versioning, management, and context-aware deployment.

### MCP Features

- **Model Registry**: Track multiple model versions with metadata
- **MCP Resources**: Read-only access to model metadata (model://registry, model://active)
- **MCP Tools**: Prediction and management operations (predict, batch_predict, compare_models)
- **Async Client**: Python async client for easy integration
- **Model Comparison**: Compare performance metrics across versions
- **Deployment Context**: Get complete context for model deployment

### Quick Start with MCP

```bash
# Terminal 1: Start MCP Server
python MCP.server.py

# Terminal 2: Use Client
python
```

```python
import asyncio
from MCP.client import FakeNewsMCPClient

async def main():
    async with FakeNewsMCPClient() as client:
        # Make a prediction
        result = await client.predict("Breaking news text here")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']}")

        # Get registry
        registry = await client.get_registry()
        print(f"Models: {registry['total_models']}")

        # Compare models
        models = await client.list_models()
        comparison = await client.compare_models([m['model_id'] for m in models['models']])
        print(comparison)

asyncio.run(main())
```

### MCP API

**Resources (Read-Only):**
- `model://registry` - Get all models and active model
- `model://active` - Get active model info
- `model://metrics/{model_id}` - Get model performance metrics
- `model://architecture/{model_id}` - Get model architecture details

**Tools (Read-Write):**
- `predict(text, model_id, return_confidence)` - Make single prediction
- `batch_predict(texts, model_id)` - Batch predictions
- `register_model_version(...)` - Register new model
- `list_all_models()` - List all models with metrics
- `set_active_model(model_id)` - Activate model for predictions
- `compare_models(model_ids)` - Compare model performance
- `get_model_context(model_id)` - Get deployment context

### MCP Documentation

- [MCP.API.md](MCP.API.md) - Complete API reference
- [MCP.example.md](MCP.example.md) - Full usage guide
- [MCP_utils.py](MCP_utils.py) - Utility functions for registry management

## Usage Examples

### Basic BERT Training

```python
from bert_utils import DataConfig, TrainingConfig, BertModelWrapper, DataLoader
from pathlib import Path

# Configure data loading
data_config = DataConfig(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    max_text_length=256,
    stratify=True
)

# Configure training
train_config = TrainingConfig(
    model_name='distilbert-base-uncased',
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=2,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    patience=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load data
loader = DataLoader()
texts, labels = loader.load_liar(Path('data/LIAR'))
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
    texts, labels, data_config
)

# Train model
model = BertModelWrapper(train_config)
history = model.train(X_train, y_train, X_val, y_val)

# Save model
model.save_model('models/bert_fake_news_detector')
```

### Model Evaluation

```python
# Evaluate on test set
from torch.utils.data import DataLoader as TorchDataLoader
from bert_utils import BertTextDataset

test_dataset = BertTextDataset(X_test, y_test, model.tokenizer, max_length=256)
test_loader = TorchDataLoader(test_dataset, batch_size=16)

metrics = model._evaluate(test_loader)

print(f"Accuracy:  {metrics.accuracy:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall:    {metrics.recall:.4f}")
print(f"F1-Score:  {metrics.f1:.4f}")
print(f"ROC-AUC:   {metrics.roc_auc:.4f}")
```

### Multi-Dataset Training

```python
# Combine multiple datasets
texts_liar, labels_liar = loader.load_liar(Path('data/LIAR'))
texts_isot, labels_isot = loader.load_isot(Path('data/ISOT'))
texts_fnn, labels_fnn = loader.load_fakenewsnet(Path('data/FakeNewsNet/combined.csv'))

texts = texts_liar + texts_isot + texts_fnn
labels = labels_liar + labels_isot + labels_fnn

# Split and train as before
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
    texts, labels, data_config
)
history = model.train(X_train, y_train, X_val, y_val)
```

## Docker Commands

### Build Docker Image
```bash
docker build -t bert-fake-news:latest .
```

### Run Jupyter Notebook Server
```bash
docker run -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  bert-fake-news:latest
```

### Train BERT Model in Container
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  bert-fake-news:latest python train_bert_liar_only.py
```

### Evaluate Model in Container
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  bert-fake-news:latest python eval_bert_liar.py
```

### Interactive Bash Shell
```bash
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  bert-fake-news:latest /bin/bash
```

## Notes

If you encounter out-of-memory issues, reduce the batch size in TrainingConfig or enable GPU with `--gpus all` flag in Docker. For port conflicts, use `docker run -p 9999:8888` to map to a different port.

## Improvements

To improve fake news detection rates, consider:

- Implementing class-weighted loss to improve fake news recall
- Using larger models like BERT-base or RoBERTa instead of DistilBERT
- Training for more epochs (5+ instead of 2)
- Creating ensembles with TF-IDF or LSTM models
- Experimenting with data augmentation or threshold optimization

## Summary

This project implements a BERT-based fake news detector trained on 58,111 samples from three datasets (LIAR, ISOT, FakeNewsNet). The system includes lazy tokenization for memory efficiency, model versioning via MCP registry, and Docker containerization. Created as part of MSML610 coursework at University of Maryland.