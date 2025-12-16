# Model Context Protocol (MCP) - REST API Framework

## What is MCP?

**MCP (Model Context Protocol)** is a standardized protocol for serving machine learning models over HTTP/REST. It provides a consistent interface for different applications to interact with ML models without needing custom integration code.

### The Problem MCP Solves

**Without Standardization:**
```
Model Developer                 Different Clients
     ↓                          ├─ Web App
  Trained Model           →     ├─ Mobile App
     ↓                          ├─ CLI Tool
  Save to disk                  ├─ Python Script
                                └─ External Services

Result: Each client writes custom code
        → Confusing, hard to maintain, inconsistent APIs
```

**With MCP:**
```
Model Developer                 Different Clients
     ↓                          ├─ Web App
  Trained Model                 ├─ Mobile App
     ↓                          ├─ CLI Tool
  MCP REST Server       →       ├─ Python Script
     ↓                          └─ External Services
  Standardized API

Result: All clients use same REST endpoints
        → Consistent, maintainable, predictable behavior
```

---

## MCP Architecture

### Core Components

```
┌─────────────────────────────────────────────────┐
│         MCP Server (e.g., port 9090)            │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │      Flask/FastAPI HTTP Server          │   │
│  └──────────────────┬──────────────────────┘   │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐   │
│  │   MCP Server Class (Standardized)       │   │
│  │                                          │   │
│  │  - predict(text)                        │   │
│  │  - predict_batch(texts)                 │   │
│  │  - list_models()                        │   │
│  │  - get_statistics()                     │   │
│  └──────────────────┬──────────────────────┘   │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐   │
│  │      Loaded ML Model(s)                 │   │
│  │                                          │   │
│  │  (BERT, XGBoost, Neural Net, etc.)      │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
         ↑
         │ HTTP/REST
         │
    ┌────┴────┬──────────────┬─────────────┐
    │          │              │             │
    ▼          ▼              ▼             ▼
  Web UI   Mobile App    Python CLI    Other Services
```

---

## Standard MCP Endpoints

MCP defines a standard set of endpoints that every server should expose:

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify server is running and ready

**Request:**
```bash
curl http://localhost:9090/health
```

**Response:**
```json
{
  "status": "healthy"
}
```

**Use Case:** Load balancing, Kubernetes probes, monitoring

---

### 2. List Models

**Endpoint:** `GET /models`

**Purpose:** Discover available models

**Response:**
```json
[
  {
    "id": "model_v1",
    "type": "classifier",
    "version": "1.0",
    "accuracy": 0.95
  }
]
```

**Use Case:** Auto-discovery, capability checking

---

### 3. Get Model Info

**Endpoint:** `GET /models/<model_id>`

**Purpose:** Retrieve model metadata and performance metrics

**Response:**
```json
{
  "id": "model_v1",
  "type": "classifier",
  "version": "1.0",
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.92,
  "parameters": 50000000,
  "training_samples": 10000
}
```

**Use Case:** Model evaluation, auditing, transparency

---

### 4. Single Prediction

**Endpoint:** `POST /predict` or `POST /api/predict`

**Purpose:** Classify/predict on a single sample

**Request:**
```json
{
  "text": "Input data here...",
  "model_id": "model_v1"
}
```

**Response:**
```json
{
  "prediction": {
    "label": "class_A",
    "confidence": 0.85
  },
  "metadata": {
    "processing_time_ms": 45
  }
}
```

**Use Case:** Web UI, interactive requests, one-off predictions

---

### 5. Batch Prediction

**Endpoint:** `POST /predict-batch`

**Purpose:** Classify/predict on multiple samples efficiently

**Request:**
```json
{
  "texts": [
    "Sample 1...",
    "Sample 2...",
    "Sample 3..."
  ],
  "model_id": "model_v1"
}
```

**Response:**
```json
{
  "total": 3,
  "predictions": [
    {"prediction": {"label": "A", "confidence": 0.85}},
    {"prediction": {"label": "B", "confidence": 0.72}},
    {"prediction": {"label": "A", "confidence": 0.91}}
  ],
  "metadata": {
    "total_processing_time_s": 0.15,
    "avg_time_per_sample_ms": 50
  }
}
```

**Use Case:** Bulk processing, data pipelines, batch jobs

---

### 6. Statistics

**Endpoint:** `GET /statistics`

**Purpose:** Server performance and usage metrics

**Response:**
```json
{
  "total_predictions": 1234,
  "uptime_seconds": 3600,
  "avg_processing_time_ms": 45,
  "min_processing_time_ms": 20,
  "max_processing_time_ms": 150
}
```

**Use Case:** Monitoring dashboards, performance tracking

---

## Standard Response Format

All MCP endpoints follow a consistent response structure:

### Success Response (200)

```json
{
  "status": "success",
  "data": { /* endpoint-specific data */ },
  "metadata": {
    "timestamp": 1702650845.234,
    "processing_time_ms": 45.32
  }
}
```

### Error Response (400/500)

```json
{
  "status": "error",
  "error": "Description of what went wrong",
  "error_code": "INVALID_INPUT"
}
```

### Common Error Codes

| Code | Status | Meaning |
|------|--------|---------|
| `MISSING_REQUIRED_FIELD` | 400 | Required JSON field missing |
| `INVALID_INPUT` | 400 | Input validation failed |
| `MODEL_NOT_FOUND` | 404 | Requested model doesn't exist |
| `SERVER_ERROR` | 500 | Internal server error |
| `MODEL_OVERLOADED` | 503 | Server busy, try again |

---

## HTTP Status Codes

| Code | Meaning | When to Use |
|------|---------|------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Client error (validation, format) |
| 404 | Not Found | Endpoint or resource doesn't exist |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Server Error | Model or server error |
| 503 | Unavailable | Server not ready/overloaded |

---

## Request/Response Examples

### Example 1: Single Prediction

**Request:**
```bash
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample input data"}'
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "label": "class_A",
    "confidence": 0.87
  },
  "metadata": {
    "processing_time_ms": 42.5,
    "input_length": 18
  }
}
```

---

### Example 2: Batch Prediction

**Request:**
```bash
curl -X POST http://localhost:9090/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Sample 1", "Sample 2", "Sample 3"]
  }'
```

**Response:**
```json
{
  "status": "success",
  "total": 3,
  "class_a_count": 2,
  "class_b_count": 1,
  "predictions": [
    {"prediction": {"label": "A", "confidence": 0.85}},
    {"prediction": {"label": "B", "confidence": 0.72}},
    {"prediction": {"label": "A", "confidence": 0.91}}
  ],
  "metadata": {
    "total_processing_time_s": 0.145,
    "avg_time_per_sample_ms": 48.3
  }
}
```

---

### Example 3: Error Handling

**Request (Invalid - missing required field):**
```bash
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response (400):**
```json
{
  "status": "error",
  "error": "Missing required field: text",
  "error_code": "MISSING_REQUIRED_FIELD"
}
```

---

## Client Integration Patterns

### Pattern 1: Python Client Library

```python
import requests

class MCPClient:
    def __init__(self, base_url='http://localhost:9090'):
        self.base_url = base_url

    def predict(self, text):
        """Single prediction."""
        response = requests.post(
            f'{self.base_url}/api/predict',
            json={'text': text}
        )
        return response.json()

    def predict_batch(self, texts):
        """Batch prediction."""
        response = requests.post(
            f'{self.base_url}/predict-batch',
            json={'texts': texts}
        )
        return response.json()

    def health_check(self):
        """Check server health."""
        response = requests.get(f'{self.base_url}/health')
        return response.json()

# Usage
client = MCPClient()
result = client.predict("Sample input")
print(result['prediction']['label'])
```

---

### Pattern 2: JavaScript/Node.js Client

```javascript
class MCPClient {
  constructor(baseUrl = 'http://localhost:9090') {
    this.baseUrl = baseUrl;
  }

  async predict(text) {
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    return await response.json();
  }

  async predictBatch(texts) {
    const response = await fetch(`${this.baseUrl}/predict-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    return await response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Usage
const client = new MCPClient();
const result = await client.predict('Sample input');
console.log(result.prediction.label);
```

---

### Pattern 3: Streaming/Generator Pattern (for large batches)

```python
def predict_large_dataset(texts, batch_size=100):
    """Process large dataset in batches."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.predict_batch(batch)

        for pred in response['predictions']:
            yield pred
```

---

## Performance Considerations

### Single vs Batch

| Aspect | Single Prediction | Batch Prediction |
|--------|------------------|------------------|
| **Latency** | ~40-50ms | ~40-50ms per item |
| **Throughput** | Low | High |
| **Best For** | Real-time, interactive | Bulk processing |
| **Network Overhead** | Low | Amortized |

### Optimization Tips

1. **Batching**: Send multiple items together
   ```python
   # Bad: N requests
   for text in texts:
       client.predict(text)

   # Good: 1 request
   client.predict_batch(texts)
   ```

2. **Connection pooling**: Reuse HTTP connections
   ```python
   session = requests.Session()
   client = MCPClient(session=session)
   ```

3. **Async requests**: Process requests concurrently
   ```python
   import asyncio
   tasks = [client.predict_async(t) for t in texts]
   results = await asyncio.gather(*tasks)
   ```

---

## Error Handling Best Practices

### Pattern: Retry with Backoff

```python
import time

def predict_with_retry(client, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.predict(text)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
```

### Pattern: Graceful Degradation

```python
def predict_with_fallback(client, text):
    try:
        return client.predict(text)
    except Exception as e:
        # Fallback behavior
        return {
            'status': 'error',
            'fallback': True,
            'prediction': {'label': 'UNKNOWN', 'confidence': 0.0},
            'error': str(e)
        }
```

---

## Deployment Models

### Standalone Server

```bash
python mcp_server.py --host 0.0.0.0 --port 9090
```

### Docker Container

```bash
docker run -p 9090:9090 mcp-server:latest
```

### Load Balanced (Multiple Instances)

```
    ┌─────────────────┐
    │  Load Balancer  │
    │   (Port 9090)   │
    └────────┬────────┘
             │
        ┌────┼────┐
        │    │    │
        ▼    ▼    ▼
      MCP  MCP  MCP
      S1   S2   S3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mcp
        image: mcp-server:latest
        ports:
        - containerPort: 9090
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 9090
```

---

## Configuration

Standard environment variables for MCP servers:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `0.0.0.0` | Server bind address |
| `MCP_PORT` | `9090` | Server port |
| `MCP_DEBUG` | `false` | Enable debug logging |
| `MCP_LOG_LEVEL` | `INFO` | Logging verbosity |
| `MCP_WORKERS` | `4` | Number of worker threads |
| `MCP_TIMEOUT` | `30` | Request timeout (seconds) |

---

## Monitoring & Observability

### Metrics to Track

```
GET /statistics

Response includes:
- total_predictions: Total requests processed
- avg_processing_time_ms: Average latency
- min_processing_time_ms: Minimum latency
- max_processing_time_ms: Maximum latency
- error_count: Failed requests
```

### Logging

Log all requests with:
- Timestamp
- Method and endpoint
- Request payload size
- Response status
- Processing time

### Health Checks

```bash
# Every 10 seconds
curl http://localhost:9090/health

# Alert if not healthy
```

---

## Authentication & Security

### Basic Authentication

```python
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    return username == 'user' and password == 'pass'

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    # Handle prediction
    pass
```

### API Key Authentication

```python
def check_api_key(request):
    key = request.headers.get('X-API-Key')
    return key == os.getenv('MCP_API_KEY')

@app.route('/predict', methods=['POST'])
def predict():
    if not check_api_key(request):
        return {'error': 'Invalid API key'}, 401
    # Handle prediction
```

### Rate Limiting

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per hour")
def predict():
    # Handle prediction
    pass
```

---

## Testing

### Unit Test Example

```python
import pytest
from mcp_server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_predict_success(client):
    response = client.post('/api/predict',
        json={'text': 'Sample input'})
    assert response.status_code == 200
    assert 'prediction' in response.json

def test_predict_missing_field(client):
    response = client.post('/api/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.json
```

---

## Best Practices

### Server Implementation

✅ **DO:**
- Implement all standard endpoints
- Use consistent response format
- Include detailed error messages
- Log all requests and errors
- Add request validation
- Use appropriate HTTP status codes

❌ **DON'T:**
- Change response format between versions
- Silently fail requests
- Return 200 for errors
- Overload single endpoints
- Skip input validation

### Client Usage

✅ **DO:**
- Check HTTP status codes
- Implement retry logic
- Batch requests when possible
- Set reasonable timeouts
- Handle errors gracefully
- Validate responses

❌ **DON'T:**
- Ignore error responses
- Assume requests succeed
- Send unbounded batches
- Retry indefinitely
- Panic on occasional failures

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Standardized ML model serving |
| **Protocol** | HTTP/REST |
| **Core Endpoints** | 6 (health, models, predict, predict-batch, statistics, model-info) |
| **Response Format** | JSON with status codes |
| **Error Handling** | HTTP codes + error messages |
| **Authentication** | Optional (API key, basic auth) |
| **Deployment** | Docker, Kubernetes, standalone |

---

## References

- [REST API Best Practices](https://restfulapi.net/)
- [HTTP Status Codes](https://httpwg.org/specs/rfc7231.html#status.codes)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Last Updated: 2025-12-15*
*Version: 1.0*
