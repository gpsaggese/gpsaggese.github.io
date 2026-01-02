# Frontend - Fake News Detector Web Application

This folder contains the web application components for the Fake News Detector, including the Streamlit user interface and Flask REST API backend.

## 📂 Files in This Directory

### 1. `app.py` (202 lines)

**Streamlit Web Interface**

The main user-facing application that provides an interactive interface for fake news detection.

**File Structure**:

```python
Lines 1-10:   Imports and module docstring
Lines 12-17:  Page configuration (title, icon, layout)
Lines 20-47:  Custom CSS for dark theme and styling
Lines 50-52:  API configuration (backend URL on port 5001)
Lines 54-60:  check_backend_health() - Verifies Flask server is running
Lines 62-78:  predict_news() - Sends text to API and handles response
Lines 81-101: Pre-loaded example dictionaries (TRUE_NEWS, FAKE_NEWS)
Lines 103-104: Page header and caption
Lines 106-131: Dropdown menus with smart reset logic
Lines 135-141: Text input area
Lines 144-149: Analyze and Clear buttons
Lines 152-202: Results display (prediction, confidence, probabilities)
```

**Key Functions**:

- **`check_backend_health()`**: Tests if Flask backend is accessible

  - Returns `True` if backend responds, `False` otherwise
  - Prevents errors when backend is down

- **`predict_news(text)`**: Sends news article to backend for classification
  - Parameters: `text` (string) - News article to analyze
  - Returns: `(result_dict, error_message)` tuple
  - Handles connection errors and timeouts

**Session State Management**:

```python
st.session_state['selected_true']  # True News dropdown index (0 = "Select...")
st.session_state['selected_fake']  # Fake News dropdown index (0 = "Select...")
st.session_state['example_text']   # Currently loaded example text
```

**Dropdown Reset Logic**:
When user selects from True News dropdown → Fake News dropdown resets to "Select..."  
When user selects from Fake News dropdown → True News dropdown resets to "Select..."

This prevents confusion about which category is being tested.

**Pre-loaded Examples**:

- **5 True News Examples**: Federal Reserve, NASA Discovery, Tech Earnings, Climate Summit, Medical Research
- **5 Fake News Examples**: Chocolate Planet, Miracle Pill, Celebrity Plot, Ancient Aliens, Free Money

**UI Components**:

1. Two dropdowns (True News / Fake News examples)
2. Text area for custom input (150px height, minimum 50 characters)
3. Two equal-width buttons (Analyze / Clear)
4. Results section showing:
   - Prediction box (green for True, red/pink for Fake)
   - Confidence percentage
   - Probability distribution with progress bars

---

### 2. `backend.py` (120 lines)

**Flask REST API Server**

The backend server that loads the PEFT RoBERTa model and handles inference requests.

**File Structure**:

```python
Lines 1-9:    Imports (Flask, PyTorch, Transformers, PEFT)
Lines 11-13:  Flask app initialization with CORS
Lines 15-17:  Global variables (model, tokenizer, device)
Lines 19-55:  load_model() - Loads RoBERTa + LoRA adapters
Lines 57-92:  predict_sentiment() - Tokenizes and runs inference
Lines 94-99:  GET / - Root endpoint with API info
Lines 101-106: GET /api/health - Health check endpoint
Lines 108-120: POST /api/predict - Main prediction endpoint
Lines 122-125: Server startup (port 5001)
```

**Key Functions**:

**`load_model()`** (lines 19-55):

- Locates checkpoint-800 using absolute path resolution
- Loads RoBERTa-base model
- Loads LoRA adapters from checkpoint
- Falls back to base RoBERTa if checkpoint not found
- Sets model to evaluation mode
- Returns: `(model, tokenizer, device)` tuple

**Path Resolution**:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "../Code_files/roberta_lora_results/checkpoint-800")
```

Uses absolute paths to avoid "file not found" errors regardless of where script is run from.

**`predict_sentiment(text)`** (lines 57-92):

- Tokenizes input text (max 512 tokens, truncation enabled)
- Runs forward pass through model
- Applies softmax to get probabilities
- Extracts confidence scores
- Maps predictions to labels
- Parameters:
  - `text` (string): News article to classify
- Returns: Dictionary with prediction, confidence, probabilities, warning

**Critical Label Mapping** (lines 77-81):

```python
# IMPORTANT: Label mapping from training data
# Label 0 = True News (from PEFT_Sentiment_Analysis_utils.py line 68)
# Label 1 = Fake News (from PEFT_Sentiment_Analysis_utils.py line 67)

true_prob = probabilities[0][0].item() * 100   # Index 0 = True News
fake_prob = probabilities[0][1].item() * 100   # Index 1 = Fake News

predicted_class = torch.argmax(logits, dim=-1).item()
prediction = "Fake News" if predicted_class == 1 else "True News"
```

This mapping MUST match the training data labels defined in `../Code_files/PEFT_Sentiment_Analysis_utils.py`:

```python
fake_df["label"] = 1  # Line 67
true_df["label"] = 0  # Line 68
```

**Confidence Warnings** (lines 83-89):

- High confidence: ≥ 80%
- Moderate confidence: 60-79%
- Low confidence: < 60%

**API Endpoints**:

1. **GET /** - API information

   - Returns: Welcome message and available endpoints

2. **GET /api/health** - Health check

   - Returns: `{"status": "healthy", "model_loaded": true}`
   - Use to verify backend is running before making predictions

3. **POST /api/predict** - Main prediction endpoint
   - Request body: `{"text": "article text here..."}`
   - Validates text is present and ≥50 characters
   - Returns:
     ```json
     {
       "prediction": "True News" or "Fake News",
       "confidence": 85.42,
       "true_probability": 85.42,
       "fake_probability": 14.58,
       "warning": "High confidence prediction"
     }
     ```
   - Error responses:
     - 400: Missing or too short text
     - 500: Prediction failure

**Port Configuration**:

- Runs on port **5001** (not 5000)
- Reason: macOS AirPlay Receiver uses port 5000 by default
- Avoids "Address already in use" errors

---

### 3. `start.sh`

**Startup Script**

Convenience script to launch both backend and frontend servers simultaneously.

**What it does**:

1. Starts Flask backend (`python backend.py`) in background
2. Waits for backend to initialize
3. Starts Streamlit frontend (`streamlit run app.py`)
4. Opens browser automatically to http://localhost:8501
5. Shows instructions for stopping servers

**Usage**:

```bash
chmod +x start.sh  # Make executable (first time only)
./start.sh         # Run both servers
```

**Note**: Currently may need manual implementation. Alternative is to run both servers in separate terminals.

---

### 4. `README.md` (this file)

**Frontend Documentation**

Describes all files in the frontend folder and how to run the application.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- pip (latest version)

### Install Dependencies

Navigate to this directory and install required packages:

```bash
cd frontend
pip install streamlit flask flask-cors torch transformers peft requests
```

## ▶️ Running the Application

### Method 1: Manual Start (Recommended for Development)

**Terminal 1 - Start Backend**:

```bash
python backend.py
```

Wait for confirmation:

```
Loading model and tokenizer...
✅ Model loaded successfully!
 * Running on http://127.0.0.1:5001
```

**Terminal 2 - Start Frontend**:

```bash
streamlit run app.py
```

Browser should automatically open to http://localhost:8501

### Method 2: Using start.sh

```bash
chmod +x start.sh
./start.sh
```

### Verify Both Servers Are Running

Test backend:

```bash
curl http://localhost:5001/api/health
# Should return: {"model_loaded":true,"status":"healthy"}
```

Access frontend:

```
Open browser to http://localhost:8501
```

## 🧪 Testing the Application

### 1. Using the Web Interface

1. Open http://localhost:8501 in browser
2. Select an example from either dropdown:
   - **True News Examples**: Federal Reserve, NASA Discovery, Tech Earnings, etc.
   - **Fake News Examples**: Chocolate Planet, Miracle Pill, Celebrity Plot, etc.
3. Click "🔍 Analyze" button
4. View results (prediction, confidence, probabilities)
5. Try "Clear" button or select different example

### 2. Testing the API Directly

**Test with cURL**:

```bash
# True News example
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve announced today that it will maintain current interest rates at their existing levels. The decision was made following a review of economic indicators including inflation rates and employment figures."}'

# Expected: {"prediction": "True News", "confidence": ~53, ...}
```

```bash
# Fake News example
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING: Famous celebrities caught in secret underground meeting planning to control the world economy. Leaked documents reveal their sinister plan."}'

# Expected: {"prediction": "Fake News", "confidence": ~78, ...}
```

**Test with Python**:

```python
import requests

url = "http://localhost:5001/api/predict"
data = {"text": "Your news article here (minimum 50 characters)..."}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
print(f"True: {result['true_probability']}%")
print(f"Fake: {result['fake_probability']}%")
```

## 🔧 Troubleshooting

### Backend Won't Start

**Problem**: `FileNotFoundError` for model checkpoint

**Solution**: Verify checkpoint exists

```bash
ls ../Code_files/roberta_lora_results/checkpoint-800/adapter_model.safetensors
```

Ensure you're running from the `frontend/` directory.

---

**Problem**: `OSError: [Errno 48] Address already in use`

**Solution**: Port 5001 is occupied

```bash
lsof -i :5001        # Find process using port
kill -9 <PID>        # Kill that process
python backend.py    # Restart
```

---

### Frontend Can't Connect to Backend

**Problem**: "❌ Cannot connect to backend"

**Solution**:

1. Verify backend is running: `curl http://localhost:5001/api/health`
2. If not running, start it: `python backend.py`
3. Check firewall isn't blocking localhost connections

---

### Dropdown Not Resetting

**Problem**: Both dropdowns stay selected

**Solution**: Current version has this fixed. Verify `app.py` has session state management:

```python
if 'selected_true' not in st.session_state:
    st.session_state['selected_true'] = 0
```

---

### Module Not Found Errors

**Problem**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**: Install dependencies

```bash
pip install streamlit flask flask-cors torch transformers peft requests
```

## 🛑 Stopping the Application

### Stop Manually

Press `Ctrl+C` in each terminal window

### Stop via Command Line

```bash
pkill -f "python.*backend.py"    # Stop Flask backend
pkill -f "streamlit"              # Stop Streamlit frontend
```

## 📡 API Reference

### Backend Endpoints

**Base URL**: `http://localhost:5001`

| Endpoint       | Method | Description   | Request                  | Response                                      |
| -------------- | ------ | ------------- | ------------------------ | --------------------------------------------- |
| `/`            | GET    | API info      | -                        | Welcome message                               |
| `/api/health`  | GET    | Health check  | -                        | `{"status": "healthy", "model_loaded": true}` |
| `/api/predict` | POST   | Classify news | `{"text": "article..."}` | Prediction with probabilities                 |

### Request/Response Examples

**Health Check**:

```bash
GET /api/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Prediction Request**:

```bash
POST /api/predict
Content-Type: application/json

{
  "text": "News article text here (minimum 50 characters required)..."
}
```

**Prediction Response**:

```json
{
  "prediction": "True News",
  "confidence": 85.42,
  "true_probability": 85.42,
  "fake_probability": 14.58,
  "warning": "High confidence prediction"
}
```

**Error Response** (text too short):

```json
{
  "error": "Text must be at least 50 characters"
}
```

## 📝 Technical Details

### Session State Variables (app.py)

- `selected_true`: Index of True News dropdown (0 means "Select...")
- `selected_fake`: Index of Fake News dropdown (0 means "Select...")
- `example_text`: Currently loaded example text

### Port Configuration

- **Backend (Flask)**: Port 5001
  - Why not 5000? macOS AirPlay Receiver uses port 5000
- **Frontend (Streamlit)**: Port 8501 (default)

### Model Loading

- Model path: `../Code_files/roberta_lora_results/checkpoint-800/`
- Uses absolute path resolution to avoid "file not found" errors
- Falls back to base RoBERTa if checkpoint missing

### Label Mapping (CRITICAL)

```
Label 0 = True News
Label 1 = Fake News
```

This mapping is defined in training data and must be consistent throughout.

---

## 📚 Related Documentation

- **Project Overview**: See main README in parent directory
- **Model Training**: See `../Code_files/README.md`
- **Training Notebook**: See `../Code_files/PEFT_Sentiment_Analysis.API.ipynb`

---

**For complete project documentation including model architecture, training process, design decisions, and evaluation metrics, refer to the main project README.**
