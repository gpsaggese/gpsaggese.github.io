# FastAPI Deployment Documentation

## Overview

The FastAPI server provides production-ready REST endpoints for fake news detection inference with optional confidence scoring and ensemble predictions.

**Key Features:**
- Single and batch prediction endpoints
- Confidence scoring with multiple methods
- Ensemble model support
- Health checks and model info
- Fast inference with GPU acceleration

## Quick Start

### Start the Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
python api_server.py

# Server running at http://localhost:8000
```

### Using the API

#### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ensemble_available": false
}
```

#### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news: New policy announced by government",
    "use_confidence": true,
    "threshold": 0.5
  }'
```

Response:
```json
{
  "prediction": 1,
  "prediction_label": "Real",
  "probability": 0.82,
  "confidence": 0.82,
  "threshold": 0.5
}
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ensemble_available": false
}
```

### 2. Model Information

**GET** `/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_name": "distilbert-base-uncased",
  "model_type": "transformer",
  "num_labels": 2,
  "max_text_length": 256,
  "device": "cuda",
  "ensemble_enabled": false
}
```

### 3. Single Prediction

**POST** `/predict`

Make a single prediction for a text.

**Request:**
```json
{
  "text": "Article text to classify",
  "use_confidence": true,
  "use_ensemble": false,
  "threshold": 0.5
}
```

**Parameters:**
- `text` (str, required): Text to classify
- `use_confidence` (bool): Include confidence score
- `use_ensemble` (bool): Use ensemble model if available
- `threshold` (float): Decision threshold (0.0-1.0)

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Real",
  "probability": 0.82,
  "confidence": 0.82,
  "threshold": 0.5
}
```

### 4. Batch Prediction

**POST** `/predict_batch`

Make predictions for multiple texts efficiently.

**Request:**
```json
{
  "texts": [
    "First article text",
    "Second article text",
    "Third article text"
  ],
  "use_confidence": true,
  "use_ensemble": false
}
```

**Parameters:**
- `texts` (list, required): List of texts to classify
- `use_confidence` (bool): Include confidence scores
- `use_ensemble` (bool): Use ensemble model if available

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "prediction_label": "Real",
      "probability": 0.82,
      "confidence": 0.82,
      "threshold": 0.5
    },
    {
      "prediction": 0,
      "prediction_label": "Fake",
      "probability": 0.68,
      "confidence": 0.68,
      "threshold": 0.5
    }
  ],
  "total": 2,
  "processing_time_ms": 245.3
}
```

### 5. Confidence Analysis

**POST** `/confidence`

Get detailed confidence analysis for a prediction.

**Request:**
```json
{
  "text": "Article text",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "text": "Article text",
  "prediction": 1,
  "probability": 0.82,
  "confidence": 0.82,
  "high_confidence": true,
  "method": "probability"
}
```

### 6. Model Metrics

**GET** `/metrics`

Get model evaluation metrics from last training.

**Response:**
```json
{
  "accuracy": 0.6234,
  "precision": 0.6156,
  "recall": 0.6234,
  "f1_score": 0.6193,
  "roc_auc": 0.6234
}
```

## Usage Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "text": "Breaking news: Major discovery announced",
        "use_confidence": True
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    f"{BASE_URL}/predict_batch",
    json={
        "texts": [
            "Article 1",
            "Article 2",
            "Article 3"
        ],
        "use_confidence": True
    }
)
batch_result = response.json()
for pred in batch_result['predictions']:
    print(f"{pred['prediction_label']} ({pred['confidence']:.1%})")
```

### JavaScript/Node.js Client

```javascript
const BASE_URL = "http://localhost:8000";

async function predict(text) {
    const response = await fetch(`${BASE_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text,
            use_confidence: true
        })
    });
    return await response.json();
}

async function batchPredict(texts) {
    const response = await fetch(`${BASE_URL}/predict_batch`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            texts: texts,
            use_confidence: true
        })
    });
    return await response.json();
}

// Usage
const result = await predict("Article text here");
console.log(result);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text", "use_confidence": true}'

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2"],
    "use_confidence": true
  }'

# Get model info
curl http://localhost:8000/info

# Confidence analysis
curl -X POST http://localhost:8000/confidence \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text"}'
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t fake-news-api:latest .
```

### Run Container

```bash
# With GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fake-news-api:latest

# Without GPU (CPU only)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  fake-news-api:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up
```

## Performance

### Inference Speed

| Configuration | Single | Batch (32) | GPU | CPU |
|---|---|---|---|---|
| BERT only | ~100ms | ~800ms | ~50ms | ~200ms |
| With confidence | ~120ms | ~1000ms | ~60ms | ~250ms |
| Ensemble | ~300ms | ~2500ms | ~150ms | ~600ms |

### Memory Usage

| Component | Memory |
|---|---|
| BERT model | ~300 MB |
| Ensemble (BERT+TF-IDF+LSTM) | ~900 MB |
| API server overhead | ~200 MB |
| Total | ~500 MB - 1.1 GB |

## Best Practices

### 1. Use Batch Prediction for Multiple Texts
```python
# Good - batch
response = requests.post(f"{BASE_URL}/predict_batch",
    json={"texts": [text1, text2, text3]})

# Avoid - loop of single predictions
for text in [text1, text2, text3]:
    response = requests.post(f"{BASE_URL}/predict",
        json={"text": text})
```

### 2. Cache Model Predictions
```python
# Use Redis or similar for caching
cache[text_hash] = prediction
```

### 3. Handle Timeouts
```python
# Set reasonable timeouts
response = requests.post(url, json=data, timeout=30)
```

### 4. Monitor Confidence
```python
predictions = batch_result['predictions']

high_confidence = [p for p in predictions if p['confidence'] >= 0.8]
low_confidence = [p for p in predictions if p['confidence'] < 0.6]

print(f"High confidence: {len(high_confidence)}")
print(f"Low confidence (review needed): {len(low_confidence)}")
```

### 5. Use Ensemble for Critical Decisions
```python
# For important decisions, use ensemble
response = requests.post(f"{BASE_URL}/predict",
    json={
        "text": important_text,
        "use_ensemble": True,
        "use_confidence": True
    })
```

## Error Handling

### Common Errors

**503 Service Unavailable:**
```json
{"detail": "Model not loaded"}
```
**Solution:** Wait for model to load or check logs

**500 Internal Server Error:**
```json
{"detail": "Error message"}
```
**Solution:** Check API logs for detailed error

**400 Bad Request:**
```json
{"detail": "validation error"}
```
**Solution:** Check request format

## Advanced Configuration

### Custom Port
```python
# In api_server.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### CORS Support
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request, request_state):
    # Limited to 100 requests per minute
```

## Monitoring

### Access Logs
```bash
# Tail API logs
tail -f api.log
```

### Model Performance
```bash
# Check metrics
curl http://localhost:8000/metrics
```

### Request Metrics
```python
# Add Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

## Files

- `api_server.py` - FastAPI server implementation
- `API_DOCUMENTATION.md` - This document
- `Dockerfile` - Docker containerization
- `docker-compose.yml` - Docker Compose configuration

## Next Steps

1. Start the API server
2. Test endpoints using provided examples
3. Integrate with your application
4. Monitor performance and adjust as needed
5. Move to Task 8 (Unit Testing)

## References

- FastAPI docs: https://fastapi.tiangolo.com/
- Uvicorn docs: https://www.uvicorn.org/
- Docker docs: https://docs.docker.com/
- REST API best practices: https://restfulapi.net/
