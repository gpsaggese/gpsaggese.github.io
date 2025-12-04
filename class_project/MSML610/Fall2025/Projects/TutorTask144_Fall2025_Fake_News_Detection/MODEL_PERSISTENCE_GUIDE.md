# Model Persistence Guide

## Why Model Persistence Matters

When you build a web application for fake news detection, you **CANNOT retrain the model every time** the web server starts. This guide explains the complete model persistence system and how to use it.

---

## The Problem: Training Takes Hours!

**Without Model Persistence:**
- User visits webpage → Flask/Django starts → Model trains (2-4 hours) → User waits forever ❌

**With Model Persistence:**
- Model trained ONCE → Saved to disk → Web server loads in seconds → User gets predictions immediately ✅

---

## What We've Implemented

### 3 New Persistence Systems

#### 1. **model_persistence.py** - Universal Model Manager
Generic system for saving/loading any model type with metadata

#### 2. **bert_utils.py Enhancement** - BERT Model Persistence
```python
# Save a trained BERT model
model_id = bert_model.save_model(
    model_name="distilbert_v1",
    metrics={'accuracy': 0.85, 'f1': 0.82},
    save_dir="./models"
)
# Returns: "distilbert_v1_20251203_143022"

# Load the saved model later
bert_model, metadata = BertModelWrapper.load_model(
    model_id="distilbert_v1_20251203_143022",
    device='cpu'
)
```

#### 3. **lstm_utils.py Enhancement** - LSTM Model Persistence
```python
# Save a trained LSTM model
model_id = lstm_model.save_model(
    model_name="lstm_v1",
    metrics={'accuracy': 0.82, 'f1': 0.80},
    save_dir="./models"
)

# Load the saved model
lstm_model, metadata = LSTMModelWrapper.load_model(
    model_id="lstm_v1_20251203_143022",
    device='cpu'
)
```

---

## Directory Structure When Models are Saved

```
models/
├── bert_v1_20251203_143022/
│   ├── model/                 ← BERT model weights
│   ├── tokenizer/             ← BERT tokenizer
│   ├── metadata.json          ← Model info & metrics
│   └── config.json            ← Training configuration
│
├── lstm_v1_20251203_143022/
│   ├── lstm_model.pt          ← Full LSTM model
│   ├── lstm_weights.pt        ← LSTM weights only
│   ├── vocab.pkl              ← Vocabulary dictionary
│   ├── config.json            ← LSTM configuration
│   └── metadata.json          ← Model info & metrics
│
└── model_registry.json        ← Index of all saved models
```

---

## What Gets Saved for Each Model Type

### BERT Model
- **model/**: HuggingFace transformer model weights
- **tokenizer/**: HuggingFace tokenizer files
- **metadata.json**: Model info, accuracy, training config, creation time
- **config.json**: Training parameters (batch size, learning rate, etc.)

### LSTM Model
- **lstm_model.pt**: Full PyTorch model (recommended)
- **lstm_weights.pt**: Just the weights (smaller file size)
- **vocab.pkl**: Word vocabulary dictionary
- **metadata.json**: Model info, accuracy, metrics
- **config.json**: LSTM configuration

---

## Quick Start: Train, Save, and Load

### Step 1: Train and Save (One Time)

```python
from bert_utils import BertModelWrapper, TrainingConfig

# Create and train model
config = TrainingConfig(num_epochs=3)
bert = BertModelWrapper(config)
history = bert.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = bert.evaluate(X_test, y_test)

# SAVE THE MODEL
model_id = bert.save_model(
    model_name="fake_news_detector_v1",
    metrics=metrics,
    save_dir="./models"
)

print(f"Model saved as: {model_id}")
# Output: Model saved as: fake_news_detector_v1_20251203_143022
```

### Step 2: Load and Use (Infinite Times)

```python
from bert_utils import BertModelWrapper

# Load the saved model
bert, metadata = BertModelWrapper.load_model(
    model_id="fake_news_detector_v1_20251203_143022",
    device='cpu'
)

# Use immediately for predictions
predictions, scores = bert.predict_with_threshold(
    texts=["Breaking news article...", "Another article..."],
    threshold=0.5
)

print(predictions)  # [0, 1] → [Fake, Real]
```

---

## Web Application Integration

### Example: Flask Web App

```python
from flask import Flask, request, jsonify
from bert_utils import BertModelWrapper

app = Flask(__name__)

# Load model ONCE when app starts
MODEL_ID = "fake_news_detector_v1_20251203_143022"
model, metadata = BertModelWrapper.load_model(MODEL_ID, device='cpu')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Web endpoint for predictions.
    No model training required!
    """
    data = request.json
    text = data.get('text')
    
    # Make prediction using saved model
    predictions, scores = model.predict_with_threshold([text])
    
    return jsonify({
        'prediction': predictions[0],
        'confidence': scores[0],
        'label': 'Fake' if predictions[0] == 0 else 'Real'
    })

if __name__ == '__main__':
    app.run(port=5000)
```

**Timeline:**
- Flask startup: ~2 seconds ✅
- First prediction: ~100-500ms ✅
- Subsequent predictions: ~100-500ms each ✅

---

## Listing and Managing Saved Models

### View All Saved Models

```python
from model_persistence import ModelPersistence

persistence = ModelPersistence(models_dir="./models")

# Print registry
persistence.print_registry()

# Output:
# ================================================================================
# MODEL REGISTRY
# ================================================================================
#
# Model ID: fake_news_detector_v1_20251203_143022
#   Type:       bert
#   Created:    20251203_143022
#   Accuracy:   0.8512
#   F1:         0.8231
#   Exists:     Yes
#
# Model ID: fake_news_detector_v2_20251203_150000
#   Type:       bert
#   Created:    20251203_150000
#   Accuracy:   0.8687  ← Better version!
#   F1:         0.8456
#   Exists:     Yes
```

### Get Best Model

```python
best_model_id = persistence.get_best_model(metric='accuracy')
# Output: 'fake_news_detector_v2_20251203_150000'

# Load and use best model
model, metadata = BertModelWrapper.load_model(best_model_id)
```

### Delete Old Models

```python
# Remove old/poor performing models to save disk space
persistence.delete_model("fake_news_detector_v1_20251203_143022")
```

---

## Use Cases for Web Deployment

### 1. Inference-Only Web Service

```python
# Load best model once
model, _ = BertModelWrapper.load_model(best_model_id)

# Service millions of predictions without retraining
while True:
    user_text = input("Enter text to classify: ")
    pred, score = model.predict(user_text)
    print(f"Prediction: {pred}, Confidence: {score}")
```

### 2. A/B Testing Multiple Models

```python
# Load two versions
model_v1, _ = BertModelWrapper.load_model("model_v1_id")
model_v2, _ = BertModelWrapper.load_model("model_v2_id")

# Route users between them
if user_id % 2 == 0:
    pred, score = model_v1.predict(text)
    version = "v1"
else:
    pred, score = model_v2.predict(text)
    version = "v2"
```

### 3. Model Versioning and Rollback

```python
# If v2 performs poorly, revert to v1 instantly
# No retraining needed!
best_model, _ = BertModelWrapper.load_model("model_v1_id")

# Switch back at runtime
best_model, _ = BertModelWrapper.load_model("model_v2_id")
```

---

## Metadata Stored with Each Model

Each saved model includes metadata.json with:

```json
{
  "model_id": "fake_news_detector_v1_20251203_143022",
  "model_name": "fake_news_detector_v1",
  "model_type": "bert",
  "created_at": "20251203_143022",
  "config": {
    "model_name": "distilbert-base-uncased",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3
  },
  "metrics": {
    "accuracy": 0.8512,
    "precision": 0.8634,
    "recall": 0.8231,
    "f1": 0.8231,
    "roc_auc": 0.92
  },
  "training_history": {
    "train_loss": [...],
    "val_accuracy": [...]
  }
}
```

Use this metadata to:
- Compare model versions
- Track accuracy improvements
- Reproduce training configuration
- Document model performance

---

## Performance Benefits

### Without Model Persistence
```
Web Request
   ↓
Load model from HuggingFace (download ~400MB)
   ↓
Train model on data (2-4 hours)
   ↓
Make prediction (100ms)
   ↓
Response to user (4+ hours later) ❌
```

### With Model Persistence
```
Server Startup (once)
   ↓
Load saved model from disk (2 seconds)
   ↓
Web Request #1
   ↓
Make prediction (100ms)
   ↓
Response to user (100ms later) ✅

Web Request #2
   ↓
Make prediction (100ms)
   ↓
Response to user (100ms later) ✅
```

---

## Storage Requirements

### Model Sizes

| Model Type | Size | Load Time |
|---|---|---|
| DistilBERT | ~270 MB | ~1-2 sec |
| BERT-base | ~440 MB | ~2-3 sec |
| LSTM | ~50-100 MB | <1 sec |

### Docker Deployment

With model persistence in Docker:
- Base image: ~500 MB
- Pre-saved models: ~500 MB
- **Total**: ~1 GB
- **Startup time**: ~5 seconds
- **First prediction**: ~100ms

---

## Best Practices

### 1. Save Models with Descriptive Names
```python
# Good ✅
model_id = bert.save_model("distilbert_liar_v1_balanced_classes")

# Bad ❌
model_id = bert.save_model("model1")
```

### 2. Always Save Metrics
```python
# Always include metrics for comparison
model_id = bert.save_model(
    "my_model",
    metrics={
        'accuracy': 0.85,
        'f1': 0.82,
        'precision': 0.86,
        'recall': 0.79
    }
)
```

### 3. Version Your Models
```python
# Use version numbers for tracking
model_id1 = bert.save_model("fake_news_detector_v1")
model_id2 = bert.save_model("fake_news_detector_v2")  # With improvements
model_id3 = bert.save_model("fake_news_detector_v3")  # Even better
```

### 4. Document Model Selection
```python
persistence = ModelPersistence()
best_id = persistence.get_best_model('accuracy')

# Load and use
model, metadata = BertModelWrapper.load_model(best_id)

# Log which model is running
print(f"Using model: {metadata['model_name']}")
print(f"Accuracy: {metadata['metrics']['accuracy']}")
```

---

## Summary

✅ **Train Once, Use Forever**
- Save models after training
- Load instantly for predictions
- No retraining needed

✅ **Perfect for Web Deployment**
- Fast server startup (<5 seconds)
- Instant predictions (100-500ms)
- Multiple models for A/B testing

✅ **Version Control**
- Compare model versions
- Track metrics over time
- Rollback if needed

✅ **Production Ready**
- Metadata stored with each model
- Error handling included
- Easy integration with Flask/Django

---

## Next Steps

1. **Train and Save Your Models**
   ```python
   model_id = bert.save_model("fake_news_v1", metrics=eval_metrics)
   ```

2. **List All Saved Models**
   ```python
   persistence.print_registry()
   ```

3. **Load Best Model for Web App**
   ```python
   best_id = persistence.get_best_model()
   model, metadata = BertModelWrapper.load_model(best_id)
   ```

4. **Deploy to Web Server**
   See section above for Flask example

Your model persistence system is ready! 🚀
