# BERT Fake News Detection - MCP API Documentation

## Overview

The BERT Fake News Detection system exposes a complete REST API through the Model Context Protocol (MCP) server. The server runs on **port 9090** and provides standardized endpoints for:

- Health monitoring
- Model information and discovery
- Single article classification
- Batch article classification
- Server statistics and metrics

---

## API Base URL

```
http://localhost:9090
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Verify that the MCP server is running and healthy

**Request:**
```bash
curl http://localhost:9090/health
```

**Response (Success - 200):**
```json
{
  "status": "healthy",
  "server": "MCP Server running"
}
```

**Response (Error - 503):**
```json
{
  "status": "unhealthy",
  "error": "Server not initialized"
}
```

**Use Cases:**
- Monitoring and load balancing
- Kubernetes readiness probes
- Health dashboards
- Server startup verification

---

### 2. List Available Models

**Endpoint:** `GET /models`

**Purpose:** Discover which models are available on the server

**Request:**
```bash
curl http://localhost:9090/models
```

**Response (Success - 200):**
```json
[
  {
    "id": "bert_fake_news",
    "type": "bert-fake-news-detector",
    "accuracy": 0.8474,
    "version": "1.0"
  }
]
```

**Use Cases:**
- Auto-discovery of available models
- Integration with client applications
- Capability checking
- Version management

---

### 3. Get Model Information

**Endpoint:** `GET /models/<model_id>`

**Purpose:** Retrieve detailed information about a specific model

**Parameters:**
- `model_id` (path): The model identifier (e.g., "bert_fake_news")

**Request:**
```bash
curl http://localhost:9090/models/bert_fake_news
```

**Response (Success - 200):**
```json
{
  "id": "bert_fake_news",
  "type": "bert-fake-news-detector",
  "version": "1.0",
  "training_accuracy": 0.9991,
  "unseen_accuracy": 0.8474,
  "precision": 0.8213,
  "recall": 0.8797,
  "f1_score": 0.8495,
  "roc_auc": 0.9360,
  "parameters": 110000000,
  "model_type": "bert-base-uncased",
  "test_samples": 6734,
  "unseen_samples": 64951,
  "gpu_acceleration": true
}
```

**Response (Error - 500):**
```json
{
  "error": "Model bert_unknown not found"
}
```

**Model Metrics Explained:**
| Metric | Value | Meaning |
|--------|-------|---------|
| `training_accuracy` | 99.91% | Accuracy on training data |
| `unseen_accuracy` | 84.74% | Accuracy on completely unseen data |
| `precision` | 82.13% | False positive rate |
| `recall` | 87.97% | False negative rate |
| `f1_score` | 84.95% | Harmonic mean of precision & recall |
| `roc_auc` | 93.60% | Area under ROC curve |
| `parameters` | 110M | Total model parameters |

**Use Cases:**
- Model evaluation and selection
- Performance transparency
- Auditing and compliance
- Client-side validation

---

### 4. Single Article Prediction (Web API)

**Endpoint:** `POST /api/predict`

**Purpose:** Classify a single news article as REAL or FAKE (recommended for web UI)

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Breaking News: Scientists announce major discovery. Researchers at the university announced today a significant breakthrough in their research."
}
```

**Response (Success - 200):**
```json
{
  "status": "success",
  "label": "REAL",
  "confidence": 0.8754,
  "confidence_percent": 87.54,
  "processing_time_ms": 45.32,
  "text_length": 145
}
```

**Response (Error - 400, Missing Text):**
```json
{
  "error": "Missing required field: text",
  "status": "error"
}
```

**Response (Error - 400, Empty Text):**
```json
{
  "error": "Text field cannot be empty",
  "status": "error"
}
```

**Response (Error - 500, Server Error):**
```json
{
  "error": "Internal server error",
  "status": "error"
}
```

**Field Descriptions:**
| Field | Type | Description |
|-------|------|-------------|
| `label` | string | "REAL" or "FAKE" |
| `confidence` | float | Confidence score (0.0 - 1.0) |
| `confidence_percent` | float | Confidence as percentage (0.0 - 100.0) |
| `processing_time_ms` | float | Time taken to process (milliseconds) |
| `text_length` | integer | Length of input text (characters) |

**curl Example:**
```bash
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Article text here..."}'
```

**Python Example:**
```python
import requests

response = requests.post(
    'http://localhost:9090/api/predict',
    json={'text': 'Your article text here...'}
)

result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence_percent']}%")
```

**JavaScript Example:**
```javascript
fetch('http://localhost:9090/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Article text here...'})
})
.then(r => r.json())
.then(data => {
  console.log(`${data.label} (${data.confidence_percent}%)`);
});
```

**Use Cases:**
- Web UI integration
- Single article classification
- User submissions
- One-off predictions

---

### 5. Single Article Prediction (MCP Standard)

**Endpoint:** `POST /predict`

**Purpose:** Classify a single article (MCP protocol compliant, returns full metadata)

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "Article text here...",
  "model_id": "bert_fake_news"
}
```

**Response (Success - 200):**
```json
{
  "model_id": "bert_fake_news",
  "text": "Article text here...",
  "prediction": {
    "label": 1,
    "class": "REAL",
    "confidence": 0.8754,
    "confidence_percent": "87.54%"
  },
  "metadata": {
    "processing_time_ms": 45.32,
    "text_length": 145,
    "timestamp": 1702650845.234
  }
}
```

**Field Descriptions:**
| Field | Description |
|-------|-------------|
| `model_id` | The model used for prediction |
| `text` | First 200 chars of input (truncated for long texts) |
| `prediction.label` | 0 = FAKE, 1 = REAL |
| `prediction.class` | "FAKE" or "REAL" |
| `prediction.confidence` | Confidence score (0.0 - 1.0) |
| `prediction.confidence_percent` | Confidence as percentage string |
| `metadata.timestamp` | Unix timestamp of prediction |

**curl Example:**
```bash
curl -X POST http://localhost:9090/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text here...",
    "model_id": "bert_fake_news"
  }'
```

**Use Cases:**
- Programmatic integration
- ML pipeline integration
- Batch processing systems
- Cross-system compatibility

---

### 6. Batch Article Prediction

**Endpoint:** `POST /predict-batch`

**Purpose:** Classify multiple articles in a single request (up to 100+ articles)

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "texts": [
    "First article text here...",
    "Second article text here...",
    "Third article text here..."
  ],
  "model_id": "bert_fake_news"
}
```

**Response (Success - 200):**
```json
{
  "model_id": "bert_fake_news",
  "total": 3,
  "real_count": 2,
  "fake_count": 1,
  "real_percent": "66.7%",
  "fake_percent": "33.3%",
  "avg_confidence": 0.8234,
  "predictions": [
    {
      "model_id": "bert_fake_news",
      "text": "First article text here...",
      "prediction": {
        "label": 1,
        "class": "REAL",
        "confidence": 0.8754,
        "confidence_percent": "87.54%"
      },
      "metadata": {
        "processing_time_ms": 45.32,
        "text_length": 125,
        "timestamp": 1702650845.234
      }
    },
    {
      "model_id": "bert_fake_news",
      "text": "Second article text here...",
      "prediction": {
        "label": 0,
        "class": "FAKE",
        "confidence": 0.7123,
        "confidence_percent": "71.23%"
      },
      "metadata": {
        "processing_time_ms": 42.15,
        "text_length": 135,
        "timestamp": 1702650845.280
      }
    },
    {
      "model_id": "bert_fake_news",
      "text": "Third article text here...",
      "prediction": {
        "label": 1,
        "class": "REAL",
        "confidence": 0.8234,
        "confidence_percent": "82.34%"
      },
      "metadata": {
        "processing_time_ms": 43.87,
        "text_length": 142,
        "timestamp": 1702650845.325
      }
    }
  ],
  "metadata": {
    "total_processing_time_s": 0.131,
    "avg_time_per_article_ms": 43.78,
    "timestamp": 1702650845.325
  }
}
```

**Response (Error - 400, Missing Texts):**
```json
{
  "error": "Missing required field: texts",
  "status": "error"
}
```

**Response (Error - 400, Invalid Format):**
```json
{
  "error": "texts must be a list",
  "status": "error"
}
```

**Field Descriptions:**
| Field | Description |
|-------|-------------|
| `total` | Number of articles classified |
| `real_count` | Count of articles classified as REAL |
| `fake_count` | Count of articles classified as FAKE |
| `real_percent` | Percentage classified as REAL |
| `fake_percent` | Percentage classified as FAKE |
| `avg_confidence` | Average confidence across all predictions |
| `predictions` | Array of individual predictions |
| `metadata.total_processing_time_s` | Total processing time (seconds) |
| `metadata.avg_time_per_article_ms` | Average time per article (milliseconds) |

**curl Example:**
```bash
curl -X POST http://localhost:9090/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Article 1 text...",
      "Article 2 text...",
      "Article 3 text..."
    ],
    "model_id": "bert_fake_news"
  }'
```

**Python Example:**
```python
import requests

texts = [
    "Article 1 text...",
    "Article 2 text...",
    "Article 3 text..."
]

response = requests.post(
    'http://localhost:9090/predict-batch',
    json={'texts': texts}
)

result = response.json()
print(f"Total: {result['total']}")
print(f"Real: {result['real_count']} ({result['real_percent']})")
print(f"Fake: {result['fake_count']} ({result['fake_percent']})")
print(f"Avg Confidence: {result['avg_confidence']:.2%}")

for pred in result['predictions']:
    label = pred['prediction']['class']
    conf = pred['prediction']['confidence_percent']
    print(f"  - {label} ({conf})")
```

**Use Cases:**
- News aggregators
- Bulk classification
- CSV/database imports
- Content moderation
- Research analysis
- Performance testing

---

### 7. Server Statistics

**Endpoint:** `GET /statistics`

**Purpose:** Retrieve server usage statistics and prediction metrics

**Request:**
```bash
curl http://localhost:9090/statistics
```

**Response (Success - 200, With Predictions):**
```json
{
  "total_predictions": 1234,
  "real_predictions": 756,
  "fake_predictions": 478,
  "avg_confidence": 0.8345,
  "min_confidence": 0.5123,
  "max_confidence": 0.9876
}
```

**Response (Success - 200, No Predictions):**
```json
{
  "total_predictions": 0,
  "message": "No predictions made yet"
}
```

**Field Descriptions:**
| Field | Description |
|-------|-------------|
| `total_predictions` | Total number of predictions made |
| `real_predictions` | Count of articles classified as REAL |
| `fake_predictions` | Count of articles classified as FAKE |
| `avg_confidence` | Average confidence across all predictions |
| `min_confidence` | Lowest confidence score |
| `max_confidence` | Highest confidence score |

**Use Cases:**
- Monitoring and dashboards
- Performance metrics
- Usage analytics
- Load balancing decisions
- Auditing and compliance

---

## Common Response Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Prediction completed successfully |
| 400 | Bad Request | Missing or invalid JSON fields |
| 404 | Not Found | Endpoint does not exist |
| 500 | Server Error | Model not loaded, processing error |
| 503 | Unavailable | Server not initialized (unhealthy) |

---

## Error Handling

### Example Error Response:

```json
{
  "error": "Text field cannot be empty",
  "status": "error"
}
```

### Best Practices:

1. Always check response status code first
2. Handle 400-level errors (client mistakes) with user feedback
3. Handle 500-level errors (server issues) with retry logic
4. Validate input before sending to server
5. Set reasonable request timeouts (e.g., 30 seconds)
6. Implement exponential backoff for retries

---

## Request/Response Examples

### Example 1: Classify a Single Article

**Task:** Determine if an article is real or fake

**Request:**
```bash
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Reuters Reports: Government Announces New Education Policy Officials confirmed today that a new comprehensive education reform will begin next year, focusing on STEM subjects and teacher training."
  }'
```

**Response:**
```json
{
  "status": "success",
  "label": "REAL",
  "confidence": 0.9234,
  "confidence_percent": 92.34,
  "processing_time_ms": 52.45,
  "text_length": 195
}
```

---

### Example 2: Batch Process News Feed

**Task:** Classify 5 articles from a news feed

**Request:**
```bash
curl -X POST http://localhost:9090/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Breaking: Company releases quarterly earnings report...",
      "SHOCKING: Celebrity reveals SECRET...",
      "Study finds link between exercise and health...",
      "UNBELIEVABLE: You wont believe what happened...",
      "Scientists publish peer-reviewed findings..."
    ]
  }'
```

**Response:**
```json
{
  "model_id": "bert_fake_news",
  "total": 5,
  "real_count": 3,
  "fake_count": 2,
  "real_percent": "60.0%",
  "fake_percent": "40.0%",
  "avg_confidence": 0.7856,
  "predictions": [
    {"prediction": {"class": "REAL", "confidence": 0.8754, "confidence_percent": "87.54%"}},
    {"prediction": {"class": "FAKE", "confidence": 0.6234, "confidence_percent": "62.34%"}},
    {"prediction": {"class": "REAL", "confidence": 0.8123, "confidence_percent": "81.23%"}},
    {"prediction": {"class": "FAKE", "confidence": 0.5847, "confidence_percent": "58.47%"}},
    {"prediction": {"class": "REAL", "confidence": 0.9234, "confidence_percent": "92.34%"}}
  ],
  "metadata": {
    "total_processing_time_s": 0.245,
    "avg_time_per_article_ms": 49.0,
    "timestamp": 1702650900.500
  }
}
```

---

### Example 3: Monitor Server Health

**Task:** Check if server is healthy and get statistics

**Request:**
```bash
# Health check
curl http://localhost:9090/health

# Get statistics
curl http://localhost:9090/statistics
```

**Response (Health):**
```json
{
  "status": "healthy",
  "server": "MCP Server running"
}
```

**Response (Statistics):**
```json
{
  "total_predictions": 5,
  "real_predictions": 3,
  "fake_predictions": 2,
  "avg_confidence": 0.7856,
  "min_confidence": 0.5847,
  "max_confidence": 0.9234
}
```

---

## Integration Examples

### Python Integration

```python
import requests

class FakeNewsDetector:
    def __init__(self, api_url='http://localhost:9090'):
        self.api_url = api_url

    def is_healthy(self):
        """Check if server is running."""
        response = requests.get(f'{self.api_url}/health')
        return response.json()['status'] == 'healthy'

    def predict(self, text):
        """Classify a single article."""
        response = requests.post(
            f'{self.api_url}/api/predict',
            json={'text': text}
        )
        data = response.json()
        return {
            'label': data['label'],
            'confidence': data['confidence_percent']
        }

    def predict_batch(self, texts):
        """Classify multiple articles."""
        response = requests.post(
            f'{self.api_url}/predict-batch',
            json={'texts': texts}
        )
        return response.json()

    def get_stats(self):
        """Get server statistics."""
        response = requests.get(f'{self.api_url}/statistics')
        return response.json()

# Usage
detector = FakeNewsDetector()

if detector.is_healthy():
    result = detector.predict("Article text here...")
    print(f"Result: {result['label']} ({result['confidence']}%)")

    stats = detector.get_stats()
    print(f"Total predictions: {stats['total_predictions']}")
```

### JavaScript Integration

```javascript
class FakeNewsDetector {
  constructor(apiUrl = 'http://localhost:9090') {
    this.apiUrl = apiUrl;
  }

  async predict(text) {
    const response = await fetch(`${this.apiUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    return await response.json();
  }

  async predictBatch(texts) {
    const response = await fetch(`${this.apiUrl}/predict-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    return await response.json();
  }

  async getStats() {
    const response = await fetch(`${this.apiUrl}/statistics`);
    return await response.json();
  }
}

// Usage
const detector = new FakeNewsDetector();

detector.predict("Article text...").then(result => {
  console.log(`${result.label} (${result.confidence_percent}%)`);
});
```

---

## Configuration

The server reads configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | 0.0.0.0 | Server host address |
| `MCP_PORT` | 9090 | Server port |
| `MCP_DEBUG` | false | Enable debug mode |

**Setting Environment Variables:**

```bash
# Bash
export MCP_PORT=9090
export MCP_HOST=localhost
python mcp_server.py

# Docker
docker run -e MCP_PORT=9090 -e MCP_HOST=0.0.0.0 bert-fake-news-api
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Avg Response Time | 40-50 ms per article |
| Batch Processing | ~40 ms per article (parallel) |
| Model Size | 440 MB (BERT base) |
| Memory Usage | ~2 GB at startup |
| GPU Acceleration | Yes (if available) |
| Max Batch Size | Unlimited (limited by memory) |

---

## Authentication & Security

Currently, the API has **no authentication**. For production deployment:

1. Add API key validation
2. Implement rate limiting
3. Use HTTPS/TLS
4. Add CORS headers if needed
5. Implement request signing
6. Add request logging and monitoring

---

## Troubleshooting

### Server won't start
- Check that port 9090 is not in use: `lsof -i :9090`
- Verify BERT model is in `models/bert_fake_news/` directory
- Check logs for errors

### Predictions are slow
- Ensure GPU is available: `nvidia-smi`
- Check server load and available memory
- Consider implementing caching

### High error rate
- Validate input text is not empty
- Check text encoding (UTF-8)
- Review server logs for specific errors

### Model accuracy is lower than expected
- Confirm correct model version is loaded
- Check if input text preprocessing matches training
- Review confidence scores (may be overly conservative)

---

## API Specification Summary

| Aspect | Details |
|--------|---------|
| **Protocol** | HTTP/REST |
| **Base URL** | `http://localhost:9090` |
| **Content Type** | JSON |
| **Authentication** | None (requires addition for production) |
| **Rate Limiting** | None (requires addition for production) |
| **Endpoints** | 7 (health, models, predict, predict-batch, statistics, get-model-info, list-models) |
| **Response Format** | JSON with status codes |
| **Error Handling** | HTTP status codes + error messages |
| **Performance** | ~45ms per prediction, batch processing supported |

---

## Additional Resources

- **Notebook Example**: `mcp_fake_news_example.ipynb` - Complete end-to-end project walkthrough
- **Server Implementation**: `mcp_server.py` - Flask REST API server
- **Server Class**: `mcp_server_class.py` - MCP server logic
- **Model Utils**: `bert_utils.py` - BERT loading and prediction utilities
- **Web UI**: `templates/index.html` - Interactive web interface

