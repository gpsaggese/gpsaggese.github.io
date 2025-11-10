# BERT API for Fake News Detection

## Overview

This API provides a lightweight, reusable wrapper layer around HuggingFace's transformers library for building BERT-based text classification models. It abstracts the complexity of BERT fine-tuning while maintaining flexibility for customization.

## Architecture

The API is structured in two layers:

### Layer 1: Contract/Interface (Abstract Components)

Defines the stable interface that any implementation must satisfy:

```python
@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    max_text_length: int = 256
    stratify: bool = True

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = 'distilbert-base-uncased'
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 1
    device: str = 'cpu'

@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    loss: float
    per_class_metrics: Dict
    confusion_matrix: Dict
```

### Layer 2: Implementation (Concrete Components)

Implements the actual BERT fine-tuning logic:

```python
class BertModelWrapper:
    """Wrapper for BERT-based text classification."""

    def __init__(self, config: TrainingConfig)
    def train(X_train, y_train, X_val, y_val) -> Dict
    def _evaluate(data_loader) -> ModelMetrics
    def save_model(path: str)
    def load_model(path: str)

class BertTextDataset(Dataset):
    """PyTorch Dataset with lazy tokenization."""

    def __init__(texts, labels, tokenizer, max_length=256)
    def __len__() -> int
    def __getitem__(idx) -> Dict

class DataLoader:
    """Utility for loading multiple fake news datasets."""

    @staticmethod
    def load_liar(data_dir: Path) -> Tuple[List[str], List[int]]
    @staticmethod
    def load_isot(data_dir: Path) -> Tuple[List[str], List[int]]
    @staticmethod
    def load_fakenewsnet(filepath: Path) -> Tuple[List[str], List[int]]
    @staticmethod
    def split_data(texts, labels, config: DataConfig) -> Tuple
```

## Core Components

### 1. Data Configuration

**Purpose:** Define data loading and preprocessing parameters

**Key Parameters:**
- `train_size`: Training set proportion (default: 0.7)
- `val_size`: Validation set proportion (default: 0.15)
- `test_size`: Test set proportion (default: 0.15)
- `max_text_length`: Maximum token length for text (default: 256)
- `stratify`: Whether to use stratified split for imbalanced classes (default: True)

**Usage:**
```python
from bert_utils import DataConfig

config = DataConfig(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    max_text_length=256
)
```

### 2. Training Configuration

**Purpose:** Define model architecture and training hyperparameters

**Key Parameters:**
- `model_name`: HuggingFace model identifier (default: 'distilbert-base-uncased')
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Optimizer learning rate (default: 2e-5)
- `num_epochs`: Number of training epochs (default: 2)
- `warmup_ratio`: Warmup steps as proportion of total (default: 0.1)
- `max_grad_norm`: Gradient clipping norm (default: 1.0)
- `patience`: Early stopping patience (default: 1)
- `device`: Torch device ('cpu' or 'cuda')

**Usage:**
```python
from bert_utils import TrainingConfig

config = TrainingConfig(
    model_name='distilbert-base-uncased',
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    device='cpu'
)
```

### 3. BertModelWrapper

**Purpose:** Main interface for BERT model training and evaluation

**Constructor:**
```python
wrapper = BertModelWrapper(config: TrainingConfig)
```

**Methods:**

#### `train(X_train, y_train, X_val, y_val) -> Dict`
Fine-tune BERT on training data with validation.

**Parameters:**
- `X_train`: List of training text samples
- `y_train`: List of training labels (0 or 1)
- `X_val`: List of validation text samples
- `y_val`: List of validation labels

**Returns:**
- Dictionary with training history (train_loss, val_loss, val_accuracy, val_f1, val_roc_auc)

**Example:**
```python
history = wrapper.train(X_train, y_train, X_val, y_val)
print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
```

#### `_evaluate(data_loader) -> ModelMetrics`
Evaluate model on a dataset.

**Parameters:**
- `data_loader`: PyTorch DataLoader

**Returns:**
- ModelMetrics instance with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix

#### `save_model(path: str)`
Save fine-tuned model to disk.

**Parameters:**
- `path`: Directory path to save model

#### `load_model(path: str)`
Load fine-tuned model from disk.

**Parameters:**
- `path`: Directory path containing saved model

### 4. BertTextDataset

**Purpose:** PyTorch Dataset with lazy tokenization for memory efficiency

**Constructor:**
```python
dataset = BertTextDataset(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_length: int = 256
)
```

**Features:**
- Lazy tokenization (tokenizes on-the-fly during batch loading)
- Automatic text pre-processing (truncation to 512 chars before tokenization)
- Consistent padding and attention masks

### 5. DataLoader Utility

**Purpose:** Multi-dataset loading support

**Methods:**

#### `load_liar(data_dir: Path) -> Tuple[List[str], List[int]]`
Load LIAR political fact-checking dataset.

#### `load_isot(data_dir: Path) -> Tuple[List[str], List[int]]`
Load ISOT news articles dataset.

#### `load_fakenewsnet(filepath: Path) -> Tuple[List[str], List[int]]`
Load FakeNewsNet combined dataset.

#### `split_data(texts, labels, config: DataConfig) -> Tuple`
Split data into train/validation/test sets with optional stratification.

**Returns:**
- Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

## Workflow

### Step 1: Configure Components
```python
from bert_utils import DataConfig, TrainingConfig

data_config = DataConfig(train_size=0.7, max_text_length=256)
train_config = TrainingConfig(model_name='distilbert-base-uncased', batch_size=16)
```

### Step 2: Load Data
```python
from bert_utils import DataLoader as BertDataLoader
from pathlib import Path

loader = BertDataLoader()
texts, labels = loader.load_liar(Path('data/LIAR'))
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
    texts, labels, data_config
)
```

### Step 3: Initialize and Train Model
```python
from bert_utils import BertModelWrapper

wrapper = BertModelWrapper(train_config)
history = wrapper.train(X_train, y_train, X_val, y_val)
```

### Step 4: Evaluate and Save
```python
# Evaluation happens during training via early stopping
wrapper.save_model('models/bert_model')
```

## Design Decisions

1. **Lazy Tokenization**: Tokenizing happens during batch loading to reduce memory footprint and avoid loading entire tokenized dataset into memory upfront.

2. **Configuration Objects**: Using dataclasses for configuration ensures type safety and makes parameters explicit and discoverable.

3. **Wrapper Pattern**: Encapsulating HuggingFace API details allows users to focus on their task without knowing the underlying library complexity.

4. **Multi-Dataset Support**: The DataLoader utility supports multiple popular fake news datasets with consistent interfaces.

5. **Early Stopping**: Built-in patience mechanism prevents overfitting and saves computational resources.

6. **Device Agnostic**: Code works on both CPU and GPU without changes.

## API Stability

The public API consists of:
- Configuration dataclasses: `DataConfig`, `TrainingConfig`, `ModelMetrics`
- Main wrapper: `BertModelWrapper`
- Utility classes: `BertTextDataset`, `DataLoader`

Changes to internal methods (prefixed with `_`) are considered breaking changes within patches.

## Dependencies

- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `pandas >= 1.0.0`
- `scikit-learn >= 1.0.0`
- `numpy >= 1.20.0`
