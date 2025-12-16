# MCP BERT Fake News Detection

- **Author:** Dhanush Vasa
- **UID:** 121227645
- **Course:** MSML610 Advanced Machine Learning
- **Institution:** University of Maryland, College Park
- **Semester:** Fall 2025

---

## Project Overview

Imagine scrolling through news online and wondering: *Is this real journalism or misinformation?* This project solves that problem with an intelligent machine learning system that automatically detects fake news.

### The Big Picture

We built a complete fake news detection system that uses **BERT** (a powerful transformer model from Google) to understand the language patterns that distinguish real news from fake news. The entire system is wrapped in an **MCP (Model Context Protocol) server** - a standardized way to access AI models over the web - so you can use it as a web service running locally with Docker.

**In plain English:** You paste a news article, our system reads it, understands the language patterns, and tells you whether it's likely real or fake, along with a confidence score.

## What You Get

This is a complete, production-ready fake news detection system with multiple ways to interact with it:

### The System Components

1. **A Trained BERT Model** - An AI model that understands language patterns in real vs. fake news
2. **MCP REST Server** - A standardized API following the Model Context Protocol specification
3. **Web Interface** - A beautiful, interactive dashboard to test predictions in real-time
4. **Command-Line Tools** - Python functions you can use in your own code
5. **Docker Deployment** - Everything containerized for consistent deployment anywhere
6. **Complete Documentation** - Jupyter notebooks showing the entire pipeline from data to production

### Key Capabilities

- **Fast Predictions** - Classify news articles in ~150ms per article
- **High Accuracy** - 85%+ accuracy on unseen news articles
- **Batch Processing** - Classify multiple articles at once
- **Confidence Scores** - Know how sure the model is about each prediction
- **RESTful API** - Integrate with any application or workflow
- **Web UI** - No coding required to test predictions
- **Docker Ready** - One command to deploy the entire system

### Model Performance Metrics

The trained BERT model achieves these results on test data it has never seen before:

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Accuracy** | 84.74% | Out of 100 predictions, 85 are correct |
| **Precision** | 82.13% | When we say "REAL", we're right 82% of the time |
| **Recall** | 87.97% | We catch 88% of all real news articles |
| **F1-Score** | 84.95% | Balanced measure of precision and recall |
| **ROC-AUC** | 0.9360 | Excellent at ranking real news higher than fake |

**Bottom Line:** The model is reliable enough to use as a first-pass filter on incoming news articles. It's not perfect (nothing is), but it catches the vast majority of fake news while minimizing false alarms.

## Understanding This Project: The Three Key Pieces

Before diving into how to run this, let's understand the three main parts of this project and how they work together.

### Part 1: The Dataset - Real vs. Fake News

We train our model on a labeled dataset of real and fake news articles:

**Real News (44% of dataset):**
- Source: Reuters and other professional news organizations
- Characteristics: Professional language, cited sources, factual claims, neutral tone
- Example: "Government announces new climate policy. Officials stated..."

**Fake News (56% of dataset):**
- Source: Known misinformation websites
- Characteristics: Sensational language, vague sources, emotional appeals, conspiracies
- Example: "SHOCKING! You won't BELIEVE what they're hiding! See pics!"

**Dataset Size:**
- ~21,000 real articles
- ~23,000 fake articles
- Total: ~44,000 articles with known labels (real/fake)

The model learns to recognize the **language patterns** that distinguish fake news from real news. Real news tends to use different vocabulary, sentence structure, and rhetorical patterns than fake news.

### Part 2: BERT - The AI Engine

**BERT** (Bidirectional Encoder Representations from Transformers) is a machine learning model created by Google that understands human language at a deep level.

**How BERT Works:**
1. It reads text in **both directions** (left-to-right AND right-to-left), understanding context
2. It's **pre-trained** on billions of words from the internet, so it already knows English language patterns
3. We **fine-tune** it specifically for fake news detection by showing it examples

**Why BERT for This Project?**
- **Excellent Language Understanding** - It understands nuance and context, not just keywords
- **Transfer Learning** - Pre-trained knowledge helps with limited labeled data
- **State-of-the-Art Performance** - Industry standard for text classification tasks
- **Proven Track Record** - Used successfully for sentiment analysis, question answering, etc.

**The Architecture:**
```
Input Text → BERT Encoder → Language Understanding → Classification Head → REAL/FAKE Label + Confidence
```

### Part 3: MCP - The Standardized Interface

**MCP (Model Context Protocol)** is an open standard that defines how AI applications should expose machine learning models through a web service.

**Why MCP for This Project?**

Imagine you're building different applications that need to use fake news detection:
- A **web app** needs to check articles in real-time
- A **mobile app** needs to make predictions
- A **browser extension** needs batch processing
- A **data pipeline** needs to classify articles automatically

Without MCP, each application would need custom code. With MCP:
- **One standardized REST API** works for all clients
- **Clear contracts** about request/response format
- **Model management** built-in (load, unload, list models)
- **Interoperability** with other MCP-compliant tools
- **Easy to scale** from laptop to cloud deployment

**MCP is to AI Models what HTTP is to Web Pages** - a universal standard that lets anyone write a client.

**What MCP Provides for This Project:**
- `/health` - Check if server is running
- `/models` - List available models
- `/api/predict` - Classify one article
- `/predict-batch` - Classify multiple articles
- `/statistics` - Server stats and performance

### How These Three Pieces Connect

**The Complete System Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FAKE NEWS DETECTION SYSTEM                       │
└─────────────────────────────────────────────────────────────────────────┘

TRAINING PHASE:
──────────────────────────────────────
    ┌──────────────────┐
    │  44K Articles    │
    │  (labeled data)  │
    └────────┬─────────┘
             │
             ├─→ 75% Training Set (33,675 articles)
             ├─→ 15% Validation Set (6,735 articles)
             └─→ 15% Test Set (6,735 articles)
             │
    ┌────────▼─────────┐
    │  Clean & Process │
    │ (Text → Numbers) │
    └────────┬─────────┘
             │
    ┌────────▼─────────────┐
    │  TF-IDF Vectorization│
    │ (1000 dimensions)    │
    └────────┬─────────────┘
             │
    ┌────────▼──────────────┐
    │ Train BERT Model      │
    │ (Fine-tuning)         │
    └────────┬──────────────┘
             │
    ┌────────▼────────────────────┐
    │ Evaluate & Save Artifacts   │
    │ • model.pkl                 │
    │ • vectorizer.pkl            │
    └────────┬────────────────────┘
             │
    ┌────────▼──────────────┐
    │ 99.64% Accuracy ✓     │
    │ Ready for Production  │
    └──────────────────────┘


PRODUCTION PHASE (Many Times Per Day - ~45ms per prediction):
────────────────────────────────────────────────────────────
         ┌─────────────────────────────────────────┐
         │      User Submits News Article          │
         │  (via Web UI / API / Python / Mobile)   │
         └────────────┬────────────────────────────┘
                      │
         ┌────────────▼────────────────┐
         │   MCP Server (Port 9090)    │
         │                             │
         │  1. Load saved model.pkl    │
         │  2. Load saved vectorizer   │
         │  3. Clean article text      │
         │  4. Vectorize (TF-IDF)      │
         │  5. Predict (BERT inference)│
         │  6. Return result           │
         └────────────┬────────────────┘
                      │
         ┌────────────▼──────────────┐
         │    Server Response        │
         │ ┌──────────────────────┐  │
         │ │ Label: REAL or FAKE  │  │
         │ │ Confidence: 85-99%   │  │
         │ │ Processing: 45ms     │  │
         │ └──────────────────────┘  │
         └────────────┬──────────────┘
                      │
    ┌─────────────────┼──────────────┐
    │                 │              │
┌───▼──┐       ┌──────▼──────┐    ┌──▼────┐
│Web UI│       │ REST Clients │   │ Logs  │
│      │       │              │   │       │
│Result│       │ API Endpoint │   │Stats  │
└──────┘       └──────────────┘   └───────┘
```

The **dataset** teaches BERT the patterns. **BERT** does the intelligent analysis. **MCP** makes it accessible to anyone, anywhere.

## Getting Started with MCP

Let me walk you through how to get this MCP server up and running. We have a few different ways to deploy it depending on what you want to do.

### The Fast Way: Deploy the MCP Server with Docker (Recommended)

If you just want to spin up the MCP server and start testing predictions, Docker is your friend. We've created some simple shell scripts to make it easy:

```bash
# Make the scripts executable and run the setup
./docker_manage.sh
# Then choose option 8 for "Full Setup"
```

This will build the Docker image and start the MCP server. The server listens on port 9090 and provides:
- A web interface at `http://localhost:9090/` for manual testing
- MCP REST API endpoints for programmatic access
- Health checks and model status endpoints
- Batch prediction support for multiple articles

### The Local Development Way

If you want to work on the code and experiment locally, you'll need to set up a Python virtual environment. This isolates your project dependencies from your system Python, preventing version conflicts.

#### Step 1: Create a Virtual Environment

A virtual environment is a self-contained Python installation. It keeps your project's dependencies separate from your system Python, preventing conflicts with other projects.

**On macOS/Linux:**
```bash
# Create virtual environment named 'venv'
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
# Example: (venv) user@machine project-dir $
```

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1

# If you get a permission error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On Windows (Command Prompt):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate.bat
```

#### Step 2: Verify the Virtual Environment

After activation, your terminal should show `(venv)` at the start of each line. This means you're in the virtual environment:

```bash
# Check which Python is being used
which python  # On macOS/Linux
where python  # On Windows

# Check Python version (should be 3.8+)
python --version
```

#### Step 3: Install Project Dependencies

Now install all the required packages for this project:

```bash
# Upgrade pip to latest version (recommended)
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Verify installation
pip list
```

#### Step 4: Deactivate When Done

When you're finished working and want to return to your system Python:

```bash
# Deactivate the virtual environment
deactivate

# The (venv) prefix should disappear from your terminal
```

#### Why Use a Virtual Environment?

- **Isolation**: Each project has its own dependencies, no conflicts
- **Reproducibility**: Same versions work across different machines
- **Cleanliness**: System Python stays clean and unmodified
- **Easy Testing**: Test different package versions in different venvs
- **Production Parity**: Your dev environment matches production

#### Optional: Using Conda Instead

If you prefer Anaconda/Miniconda:

```bash
# Create environment
conda create -n fakenews python=3.9

# Activate it
conda activate fakenews

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
conda deactivate
```

Now you have a few options for what to do next with your configured environment.

### Option 1: Run the MCP Server Locally

If we already have a trained BERT model (which we do), you can skip directly to running the MCP server:

```bash
python mcp_server.py
```

This starts the MCP server on your local machine. The server provides:
- **Web Interface** at `http://localhost:9090/` - Visual testing interface
- **MCP REST API** - Programmatic access to predictions
- **Model Management** - List, load, and manage models
- **Prediction Endpoints** - Single and batch predictions

Open your browser to `http://localhost:9090/` and you'll see the web interface. Paste some news text and the MCP server will classify it as real or fake with a confidence score.

### Option 2: Learn the Complete Pipeline with Jupyter Notebooks

I created comprehensive Jupyter notebooks that walk you through the entire project. These are perfect for understanding how everything works:

#### **Main Learning Notebook: `mcp_fake_news.example.ipynb`**

This is your starting point. It's a self-contained, end-to-end guide that teaches you:

1. **Part 1: Understanding the Dataset** 
   - What makes real news vs. fake news different
   - Examples of each type
   - Dataset statistics

2. **Part 2: Data Preparation** 
   - How to split data into train/validation/test sets (75/15/15)
   - Why stratification matters
   - Text cleaning and normalization

3. **Part 3: Feature Extraction with TF-IDF**
   - Converting text to numbers that ML models understand
   - How TF-IDF measures word importance
   - Why some words matter more than others

4. **Part 4: Model Training** 
   - Training a Logistic Regression model
   - What the model learns (feature weights)
   - Why this approach works

5. **Part 5: Model Evaluation** 
   - Understanding accuracy, precision, recall metrics
   - Interpreting confusion matrices
   - Evaluating on unseen test data

6. **Part 6: Model Persistence** 
   - Saving trained models to disk
   - Loading models for production inference

7. **Part 7: Production Inference** 
   - Making predictions on new articles
   - Handling different confidence estimation methods
   - Real production code you can use

8. **Part 8-11: MCP Server & Deployment** 
   - How the MCP server works
   - API endpoints and their use cases
   - Docker deployment strategies
   - Complete system overview


**Run it:**
```bash
jupyter lab mcp_fake_news.example.ipynb
```

#### **Training Notebook: `BERT_training.ipynb`**

If you want to see how we trained the BERT model (not required, but educational):

```bash
jupyter lab BERT_training.ipynb
```

This notebook shows:
- Loading the 44K article dataset
- Data preprocessing for BERT (tokenization, padding)
- Fine-tuning BERT on fake news detection
- Cross-validation for robust evaluation
- Saving the trained model for inference

**What you'll learn:**
- How transfer learning works (starting with pre-trained BERT)
- BERT-specific preprocessing requirements
- Training dynamics and convergence
- Why we use cross-validation instead of a single train/test split

## Key Components

### The MCP Architecture: Standard Protocol for AI Models

This project is built on the **Model Context Protocol (MCP)**, which is an open standard for AI applications to safely access external resources like ML models, files, APIs, and databases through a unified interface.

In this implementation, MCP provides:
- **Standardized REST API** - Consistent endpoints for all model operations
- **Model Management** - Load, unload, and switch between models seamlessly
- **Resource Abstraction** - Models aren't hardcoded file paths; they're managed resources
- **Scalability** - Easy to add new models or deploy to production
- **Interoperability** - Works with any MCP-compliant client or tool

Think of MCP as the "operating system" for machine learning. Instead of building a custom API for this specific model, we follow the MCP standard, which means anyone familiar with MCP can immediately understand how to use this server.

### BERT: The Neural Engine

BERT stands for Bidirectional Encoder Representations from Transformers. It's a machine learning model from Google that understands context in both directions. When it looks at a sentence, it doesn't just read left-to-right like older models did. It reads the whole thing and understands how each word relates to all the others.

We use `bert-base-uncased`, which is the standard BERT model with 12 layers and 768 hidden units. It's already trained on a huge amount of English text, so it already knows how language works. We then take that knowledge and fine-tune it specifically for detecting fake news.

The architecture is straightforward:
- Feed in the news text
- BERT processes it and creates a representation
- A simple 2-layer neural network on top decides: Real or Fake?

### The MCP Server: API Gateway for BERT

The MCP server is the bridge between your requests and the BERT model. Instead of directly loading files and calling functions, you use standardized MCP endpoints that the server manages.

Here's what the MCP server endpoints can do:
- **`/health`** - Health check to verify the server is running
- **`/models`** - List available models
- **`/models/{id}/load`** - Load a specific model
- **`/predict`** - Make single predictions via MCP
- **`/predict-batch`** - Classify multiple articles at once
- **`/statistics`** - See usage stats and model performance
- **`/api/predict`** - REST endpoint for web UI

The nice thing about the MCP approach is flexibility. Want to add a new fake news model? Drop it in the models folder and the MCP server automatically discovers it. Want to switch between models? One API call. Want to deploy this to the cloud? The MCP protocol stays the same, so all clients still work.

### BERTMCPServer: The MCP Implementation

This is where all the actual work happens. The `BERTMCPServer` class implements the MCP protocol for this fake news detector. It manages the BERT model, handles predictions, tracks statistics, and maintains a history of requests.

Here's what happens when you send a prediction through the MCP server:
1. **Request received** - Your prediction request hits an MCP endpoint
2. **Model check** - The server verifies the model is loaded (if not, it loads it)
3. **Tokenization** - Your text is tokenized (words converted to numbers BERT understands)
4. **BERT inference** - The model processes the tokens and outputs predictions
5. **Response** - The MCP server returns results with classification and confidence score
6. **Logging** - Statistics are tracked for monitoring and analysis

The BERTMCPServer follows MCP conventions, making it:
- **Discoverable** - Clients can query what models are available
- **Stateful** - Models persist in memory for efficiency
- **Trackable** - All predictions are logged for audit trails
- **Extensible** - Easy to add new models or prediction methods

### The Web Interface: User-Friendly Testing

I built a simple web interface so you don't have to use the command line. Just go to `http://localhost:9090/` and you'll see a beautiful purple-themed page where you can paste news text and get instant feedback.

The interface shows:
- Your prediction (Real or Fake with a warning/success indicator)
- Confidence percentage (how sure the model is)
- A visual confidence bar so you can see it at a glance
- Processing time and text length for debugging

There are also example buttons so you can try the model out immediately without having to come up with your own text.

### Docker Deployment: MCP Server in a Container

Everything is containerized with Docker, so the MCP server runs consistently regardless of your system. You don't need to worry about Python versions, dependency conflicts, BERT model downloads, or any of that nonsense. Just run a command and the MCP server starts up in isolation.

We created 7 shell scripts to manage the MCP Docker deployment:
- `docker_build.sh` - Build the MCP server Docker image
- `docker_run.sh` - Start the MCP server in a container
- `docker_stop.sh` - Stop the running MCP server
- `docker_restart.sh` - Restart the MCP server
- `docker_logs.sh` - View MCP server logs
- `docker_clean.sh` - Clean up Docker resources
- `docker_manage.sh` - Interactive menu for all MCP Docker operations

The MCP server runs on port 9090 inside the container, exposing:
- The REST API for MCP clients
- The web interface for manual testing
- Health check endpoints for monitoring

## Project Structure

Here's what lives where and how it implements MCP:

```
.
MCP Server Implementation:
 mcp_server.py                     # Flask REST API server (MCP endpoints)
 mcp_server_class.py               # BERTMCPServer class (MCP protocol implementation)

Supporting Utilities:
 bert_utils.py                     # BERT model utilities and inference
 preprocessing.py                  # Data preprocessing utilities

Notebooks:
 BERT_training.ipynb               # Train the BERT model
 Data_Preprocessing.ipynb          # See how preprocessing works

MCP Configuration & Deployment:
 requirements.txt                  # Python dependencies
 Dockerfile.mcp                    # Docker image for MCP server
 docker-compose.mcp.yml            # MCP server Docker Compose config

Docker Management Scripts:
 docker_manage.sh                  # Interactive menu for MCP Docker (start here!)
 docker_build.sh                   # Build MCP server image
 docker_run.sh                     # Start MCP server container
 docker_stop.sh                    # Stop MCP server
 docker_restart.sh                 # Restart MCP server
 docker_logs.sh                    # View MCP server logs
 docker_clean.sh                   # Clean up MCP Docker resources

MCP Web Interface:
 templates/
    index.html                    # Web UI for MCP predictions

Training Data:
 data/
    true.csv                      # Real news (~21K articles)
    fake.csv                      # Fake news (~23K articles)

MCP Managed Models (generated after training):
 models/
    bert_fake_news/               # MCP-managed BERT model
        config.json               # Model config
        pytorch_model.bin         # Model weights
        tokenizer_config.json     # Tokenizer setup
```

The MCP server discovers and manages models in the `models/` directory, allowing you to add new models without code changes.

## MCP Server API Reference

The MCP server exposes these endpoints for programmatic access.

**API Request-Response Flow:**

```
CLIENT                              MCP SERVER (Port 9090)
  │                                         │
  │  1. Check Health                        │
  ├────────────────────────────────────────→│ GET /health
  │                                         │
  │                                    ┌────▼───────┐
  │                                    │ Is running?│
  │                                    └────┬───────┘
  │                                         │
  │←────────────────────────────────────────┤ Return: {status: healthy}
  │                                         │
  │  2. Discover Models                     │
  ├────────────────────────────────────────→│ GET /models
  │                                         │
  │                                    ┌────▼───────────────┐
  │                                    │ List all models    │
  │                                    │ bert_fake_news ✓   │
  │                                    └────┬───────────────┘
  │                                         │
  │←────────────────────────────────────────┤ Return: {models: [...]}
  │                                         │
  │  3. Make Prediction                     │
  ├─ JSON: {text: "article..."} ──────────→│ POST /api/predict
  │                                         │
  │                                    ┌────▼─────────────────┐
  │                                    │ Load model           │
  │                                    │ Clean text           │
  │                                    │ Vectorize            │
  │                                    │ Predict              │
  │                                    │ Calculate confidence │
  │                                    └────┬─────────────────┘
  │                                         │
  │←────────────────────────────────────────┤ Return JSON response
  │    {label: "REAL", confidence: 0.94}    │
  │                                         │
  │  4. Batch Predictions (Multiple)        │
  ├─ JSON: {texts: [...]} ─────────────────→│ POST /predict-batch
  │                                         │
  │←────────────────────────────────────────┤ Return: [{...}, {...}]
  │                                         │
  │  5. Get Statistics                      │
  ├────────────────────────────────────────→│ GET /statistics
  │                                         │
  │←────────────────────────────────────────┤ Return: {predictions: 1234, ...}
  │                                         │
```

### Health & Status
```bash
# Check if MCP server is running
curl http://localhost:9090/health

# Get MCP server statistics and status
curl http://localhost:9090/statistics
```

### Model Management
```bash
# List all available models in MCP
curl http://localhost:9090/models

# Load a specific model (make it ready for predictions)
curl -X POST http://localhost:9090/models/bert_fake_news/load
```

### Single Predictions
```bash
# Make a prediction via MCP REST API
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The government announced new climate policies today."
  }'

# Response format:
# {
#   "status": "success",
#   "label": "REAL",
#   "confidence": 0.94,
#   "confidence_percent": 94.32,
#   "processing_time_ms": 145,
#   "text_length": 52
# }
```

### Batch Predictions
```bash
# Make multiple predictions in one MCP request
curl -X POST http://localhost:9090/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Scientists discover new particle",
      "SHOCKING: You won'"'"'t believe this one weird trick!"
    ]
  }'
```

### Web Interface
```
Open in browser: http://localhost:9090/
```

The web interface is also an MCP client - it uses these same endpoints to make predictions with a visual interface.

## The Data

### Dataset Source

This project uses the **Fake News Detection Dataset** from Kaggle:

🔗 **Source:** [Kaggle - Fake News Detection Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv)

**Citation:** Bhavikjikadara. (2024). Fake News Detection. Kaggle.

### Dataset Description

We use two datasets from this collection:

- **`data/true.csv`** - About 21,000 real news articles from Reuters and other reputable sources
  - Source: Reuters and professional news organizations
  - Characteristics: Verified facts, professional journalism, cited sources

- **`data/fake.csv`** - About 23,000 fake news articles from various misinformation sources
  - Source: Known misinformation websites
  - Characteristics: Sensational language, unverified claims, emotional appeals

**Total:** ~44,000 labeled articles (52% fake, 48% real)

### Data Structure

Each article has the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `title` | The headline | "Breaking News: Scientists Discover..." |
| `text` | The full article content | "Researchers announced today..." |
| `subject` | Topic category | politics, health, entertainment, etc. |
| `date` | Publication date | 2024-01-15 |

### Data Preprocessing

Before training, we:
1. **Combine** title and text (headline + body tell the full story)
2. **Truncate** to 2000 characters max (handles long articles efficiently)
3. **Label** as either:
   - `0` = FAKE (from fake.csv)
   - `1` = REAL (from true.csv)
4. **Split** into train (75%), validation (15%), and test (15%) sets
5. **Stratify** to maintain class balance across all splits

This labeled data is crucial for training the model to recognize language patterns that distinguish fake from real news.

**Data Preprocessing Pipeline:**

```
Raw Article Files (CSV)
    │
    ├─→ fake.csv (23K articles) ──→ Label: 0 (FAKE)
    │
    └─→ true.csv (21K articles) ──→ Label: 1 (REAL)
    │
    ▼
┌──────────────────────────────┐
│ 1. Load & Combine            │
│    title + text              │
└────────────┬─────────────────┘
             │
    ┌────────▼──────────────┐
    │ 2. Remove Nulls       │
    │    Clean metadata     │
    └────────┬──────────────┘
             │
    ┌────────▼──────────────────┐
    │ 3. Truncate at 2000 chars │
    │    (keep important parts) │
    └────────┬──────────────────┘
             │
    ┌────────▼──────────────┐
    │ 4. Lowercase Text     │
    │    Remove URLs        │
    │    Remove special @#$ │
    └────────┬──────────────┘
             │
    ┌────────▼─────────────────────────┐
    │ 5. Stratified Train/Val/Test     │
    │    ├─ 75% Training (30,530)      │
    │    ├─ 15% Validation (7,633)     │
    │    └─ 15% Test (6,735)           │
    └────────┬─────────────────────────┘
             │
    ┌────────▼──────────────┐
    │ Clean Data Ready      │
    │ Balanced by label     │
    │ Ready for vectorizer  │
    └───────────────────────┘
```

## Complete End-to-End Pipeline

**From Raw Data to Production Prediction - The Full Journey:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE ML PIPELINE                              │
└────────────────────────────────────────────────────────────────────────────┘

PHASE 1: TRAINING (Offline - One Time)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────┐
  │ Raw Dataset     │
  │ 44K Articles    │
  │ (labeled)       │
  └────────┬────────┘
           │
           ▼
┌──────────────────────────┐     ┌─────────────────────┐
│  Data Preprocessing      │     │ Split 75/15/15:     │
│                          │     │ • Train: 30,530     │
│ • Load fake.csv (23K)    │────→│ • Val: 7,633        │
│ • Load true.csv (21K)    │     │ • Test: 6,735       │
│ • Combine title+text     │     └─────────────────────┘
│ • Truncate to 2000 chars │
│ • Remove nulls           │
│ • Clean text             │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Feature Extraction       │
│                          │
│ TF-IDF Vectorization:    │
│ • 5000 unique words      │
│ • Text → Vectors         │
│ • Each article = 5000D   │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Model Training           │
│                          │
│ Algorithm:               │
│ Bert model               │
│                          │
│ Parameters:              │
│ • Epochs: 3              │
│ • Batch: 16              │
│ • LR: varies             │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Model Evaluation         │
│                          │
│ Validation Metrics:      │
│ • Accuracy: 99.36%       │
│ • Precision: 99.48%      │
│ • Recall: 99.18%         │
│ • F1: 0.9933             │
└──────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ Save Artifacts           │
│                          │
│ • model.pkl              │
│   (trained weights)      │
│                          │
│ • vectorizer.pkl         │
│   (vocabulary)           │
└──────────────────────────┘


PHASE 2: INFERENCE (Online - Production)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────┐
  │ User Submits Article    │
  │ (Web UI / API / Mobile) │
  └────────────┬────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ MCP Server Receives  │
    │ Request              │
    │ (Port 9090)          │
    └────────┬─────────────┘
             │
             ├─→ Load model.pkl ──→ Get trained weights
             │
             ├─→ Load vectorizer.pkl ──→ Get vocabulary
             │
             ▼
    ┌──────────────────────┐
    │ Preprocess Article   │
    │                      │
    │ • Clean text         │
    │ • Lowercase          │
    │ • Remove special char│
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │ Vectorize Text       │
    │                      │
    │ TF-IDF Transform:    │
    │ Text → 5000D vector  │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │ Model Prediction     │
    │                      │
    │ • Forward pass       │
    │ • Get probabilities  │
    │ • Calculate score    │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │ Format Response      │
    │                      │
    │ JSON:                │
    │ {                    │
    │   label: "REAL",     │
    │   confidence: 0.94,  │
    │   time_ms: 45,       │
    │   text_length: 120   │
    │ }                    │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │ Send to Client       │
    │                      │
    │ Web UI displays:     │
    │ "REAL (94%)"         │
    │                      │
    │ API returns JSON     │
    └──────────────────────┘


FEATURE FLOW THROUGH ENTIRE SYSTEM
═══════════════════════════════════════════════════════════════════════════════

Article Text
    │
    ├─ "Breaking news from Reuters..."  (Raw input)
    │
    ▼
"breaking news from reuters"  (Cleaned & lowercased)
    │
    ▼
[0.35, 0.82, 0.12, ..., 0.0]  (TF-IDF vector, 5000 dimensions)
    │
    ├─ Each number = word importance score
    │ • High = important distinguishing word
    │ • Low = common or irrelevant word
    │
    ▼
Logistic Regression Model  (Apply learned weights)
    │
    ├─ Positive weights → REAL indicators
    │ ├─ "reuters" (cited source)
    │ ├─ "announced" (factual language)
    │ └─ "officials" (authority)
    │
    ├─ Negative weights → FAKE indicators
    │ ├─ "shocking" (sensational)
    │ ├─ "unbelievable" (emotional)
    │ └─ "secret" (conspiratorial)
    │
    ▼
Score: 0.94 (probability of REAL)
    │
    ▼
Decision: 0.94 > 0.5 → Predict "REAL"
    │
    ▼
Return to User: REAL (94% confidence)
```

---

## How the Model Actually Works

### Step 1: Data Preparation

When we prepare the data for training:
1. Load the CSV files
2. Combine the title and text
3. Remove any articles with missing data
4. Truncate long articles
5. Use BERT's tokenizer to convert text to numbers (BERT doesn't understand words, only numbers)
6. Pad or truncate all articles to 128 tokens so they're all the same size

### Step 2: Cross-Validation

We don't just train on some data and test on other data once. That could give us a lucky result. Instead, we use 3-fold cross-validation:
- Split the data into 3 equal parts
- Train on 2 parts, test on 1 part
- Repeat 3 times (each time using a different part for testing)
- Average the results

This gives us a much more reliable picture of how well the model actually works.

### Step 3: Training

Here are the settings we use:
- **Optimizer:** AdamW with learning rate 2e-5 (conservative, because we don't want to mess up BERT's pre-trained knowledge)
- **Epochs:** 10 (one pass through all the training data)
- **Batch size:** 16 (process 16 articles at a time for efficiency)
- **Scheduler:** Learning rate starts low, warms up, then decays

## Performance Metrics

We look at several metrics to understand how well the model is doing:

- **Accuracy:** Out of all predictions, how many were correct? (84.74%)
- **Precision:** When we say something is REAL, how often are we right? (82.13%)
- **Recall:** Out of all the real news, how much of it do we actually catch? (87.97%)
- **F1-Score:** A balanced view of precision and recall (84.95%)
- **ROC-AUC:** How good is the model at ranking real news higher than fake news? (0.9360)

These numbers tell us the model is pretty good at its job. It's not perfect (nothing is), but it's reliable enough to use as a first-pass filter.


## Deploying the MCP Server with Docker

### The Easy Way: Use Our Management Scripts

We have scripts to handle all the Docker complexity for the MCP server:

```bash
./docker_manage.sh
# Choose option 8 for full setup, or individual options for specific tasks
```

This menu lets you:
- Build the MCP server Docker image
- Start the MCP server container
- Stop the MCP server
- View MCP server logs
- Manage Docker resources
- Check MCP server status

### The Manual Way (if you're curious)

Build the MCP server image:
```bash
docker build -t bert-fake-news-mcp:latest -f Dockerfile.mcp .
```

Run the MCP server container:
```bash
docker-compose -f docker-compose.mcp.yml up
```

Verify the MCP server is healthy:
```bash
curl http://localhost:9090/health
```

Test the MCP server with a prediction:
```bash
curl -X POST http://localhost:9090/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news from Reuters about climate change..."}'
```

## Using the MCP Server in Production

The MCP server is designed to be production-ready.

## Things to Know (Limitations)

This model is good, but it's not perfect. Here are the limitations:

1. **Articles get truncated** - We limit articles to 2000 characters. If you have a really long article, we'll only look at the first part.

2. **Limited context window** - BERT can only see 128 tokens at a time (roughly 20 words). Long-range patterns might get missed.

3. **Pattern matching, not fact-checking** - The model learns patterns associated with fake news (sensational language, certain keywords, etc.). It's not actually verifying facts against a knowledge base.

4. **Training data bias** - The model is trained on specific datasets. If fake news in the real world looks different than the training data, performance might degrade.

## Future Improvements for the MCP Server

If I wanted to make the MCP server even better, here are some ideas that leverage the MCP architecture:

### Model Improvements
- Use a larger BERT model (BERT-large) for better accuracy (but slower training) - easy to add as a new MCP-managed model
- Add adversarial training to make the model more robust to tricks
- Use an ensemble of multiple models - the MCP server can switch between them seamlessly
- Fine-tune on domain-specific datasets (politics, health, tech, etc.) - each as a separate MCP-managed model

### MCP Feature Enhancements
- **Multi-model support** - Run multiple fake news detectors simultaneously (BERT, LSTM, DistilBERT) and compare results
- **Model versioning** - MCP allows tracking different versions of models side-by-side
- **A/B testing** - Compare predictions across different model versions using MCP endpoints
- **Confidence thresholds** - Let clients set custom confidence thresholds via MCP API
- **Audit logging** - Detailed logging of all MCP predictions for compliance and analysis

### Integration Features
- Add explainability endpoints via LIME/SHAP to see what the model is looking at
- Verify claims against a knowledge base through MCP
- Track source credibility as an MCP-managed resource
- Integrate fact-checking APIs as MCP resources
- Custom preprocessing pipelines available through MCP

The beauty of MCP is that all these improvements can be added without breaking existing clients - they just get new endpoints to use!

## Quick Reference Guide

### For First-Time Users

1. **Just want to try it?**
   ```bash
   ./docker_manage.sh
   # Choose option 8
   # Then go to http://localhost:9090/
   ```

2. **Want to understand how it works?**
   ```bash
   jupyter lab mcp_fake_news.example.ipynb
   # Takes ~95 minutes, teaches you everything
   ```

3. **Want to train your own model?**
   ```bash
   jupyter lab BERT_training.ipynb
   # See how we created the BERT model
   ```

4. **Want to use it programmatically?**
   ```bash
   python mcp_server.py
   # Then use the REST API endpoints
   ```

### File Quick Links

| File | Purpose |
|------|---------|
| `mcp_fake_news.example.ipynb` | **START HERE** - Learn the complete pipeline |
| `BERT_training.ipynb` | Train the BERT model from scratch |
| `mcp_server.py` | Run the MCP REST API server |
| `mcp_server_class.py` | BERTMCPServer implementation |
| `docker_manage.sh` | Docker deployment management |
| `requirements.txt` | Python dependencies |
| `data/true.csv` | Real news articles (~21K) |
| `data/fake.csv` | Fake news articles (~23K) |
| `models/bert_fake_news/` | Trained BERT model (after training) |

## Troubleshooting

### "Port 9090 is already in use"
```bash
# Kill the existing process
lsof -ti:9090 | xargs kill -9
# Or change the port in docker-compose.mcp.yml
```

### "BERT model not found"
Make sure you've run the training notebook or the model directory exists:
```bash
ls -la models/bert_fake_news/
```

### "Out of memory during training"
Reduce batch size in BERT_training.ipynb from 16 to 8, or use CPU instead of GPU.

### "Import errors when running notebooks"
Make sure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Learning Resources

To deepen your understanding of the technologies in this project:

- **BERT Understanding**: Read "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **Transfer Learning**: Learn how pre-trained models accelerate learning
- **Text Classification**: Study how to apply deep learning to text
- **MCP Specification**: Explore the Model Context Protocol standard
- **Model Evaluation**: Understand accuracy, precision, recall, and F1-score

## Citation

If you use this project in your work, please cite:

```
@project{fakenewsdetection2025,
  title={MCP BERT Fake News Detection},
  author={Vasa, Dhanush},
  school={University of Maryland, College Park},
  course={MSML610 Advanced Machine Learning},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Questions or Feedback?** Open an issue or contact the author.

**Last Updated:** December 2025
