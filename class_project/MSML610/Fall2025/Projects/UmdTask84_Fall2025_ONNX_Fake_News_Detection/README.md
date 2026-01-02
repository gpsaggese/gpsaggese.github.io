# ONNX Fake News Detection

## Project Overview

This project implements an **end-to-end fake news detection system** with a strong emphasis on **model deployment using ONNX**.

The core idea is to:

* Train a text classification model
* Convert it to ONNX
* Validate correctness and performance
* Expose inference through a stable internal API

Two model families are explored:

* **LSTM (TensorFlow/Keras)** — primary model used for the API
* **DistilBERT (PyTorch)** — experimental model to study ONNX conversion of transformer architectures

The project is structured so that **all notebooks rely on a reusable API layer**, rather than embedding training or inference logic inline.

---

## Key Design Goals

* **Clear separation of concerns**

  * Utilities and APIs live in a Python module
  * Notebooks focus on experimentation and demonstration
* **ONNX-first deployment mindset**

  * Models are converted and evaluated in ONNX Runtime
* **Reproducibility**

  * Every notebook can be run top-to-bottom using “Restart & Run All”
* **Minimal API surface**

  * The exposed API focuses only on what a downstream user needs (e.g., `predict`)

---

## Dataset

The project uses the **Kaggle Fake and Real News Dataset**, consisting of two CSV files:

* `Fake.csv` — fake news articles
* `True.csv` — real news articles

Each sample combines:

* News title
* News article body

Labels:

* `0` → Fake
* `1` → Real

The dataset is loaded, cleaned, and shuffled through a single utility function.

---

## Project Structure

```
.
├── ONNX_Fake_News_Detection.API.md
├── ONNX_Fake_News_Detection.API.ipynb
├── ONNX_Fake_News_Detection.example.md
├── ONNX_Fake_News_Detection.example.ipynb
├── ONNX_Fake_News_Detection_utils.py
├── README.md
├── requirements.txt
├── Dockerfile
├── docker_build.sh
├── docker_bash.sh
├── docker_jupyter.sh
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── lstm_fake_news.keras
│   ├── lstm_fake_news.onnx
│   ├── lstm_tokenizer.pkl
│   ├── distilbert_fake_news.pth
│   ├── distilbert_fake_news.onnx
│   └── distilbert_fake_news.onnx.data
└── __pycache__/
```

---

## Directory and File Roles

### Core API and Utilities

* **`ONNX_Fake_News_Detection_utils.py`**
  Central implementation module containing:

  * Dataset loading
  * Model training logic
  * ONNX conversion
  * ONNX Runtime inference wrappers
  * FastAPI app factory
    All notebooks rely on this file instead of embedding logic inline.

---

### API Documentation and Usage

* **`ONNX_Fake_News_Detection.API.md`**
  Documents the internal programming interface:

  * Public functions
  * Data contracts
  * Wrapper abstractions over TensorFlow, PyTorch, and ONNX Runtime

* **`ONNX_Fake_News_Detection.API.ipynb`**
  Minimal notebook demonstrating how to use the API layer for inference and integration.

---

### End-to-End Example

* **`ONNX_Fake_News_Detection.example.md`**
  High-level narrative explaining design decisions and the end-to-end workflow.

* **`ONNX_Fake_News_Detection.example.ipynb`**
  Fully executable example covering:

  * Training
  * ONNX conversion
  * Inference
  * Runtime comparison

---

### Models

* **`models/`**
  Stores trained and converted artifacts:

  * LSTM models (`.keras`, `.onnx`)
  * Tokenizer (`.pkl`)
  * DistilBERT checkpoints (`.pth`, `.onnx`)

---

### Data

* **`data/`**
  Contains the Kaggle Fake and Real News dataset:

  * `Fake.csv`
  * `True.csv`

---

### Containerization

#### Docker Support

This project includes Docker support to ensure a **reproducible runtime environment** for training, inference, and API usage.

Docker is **optional** but recommended if you want to:

* Avoid local dependency conflicts
* Run the FastAPI service consistently
* Execute notebooks in an isolated environment

---

#### Docker Files Overview

| File                | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `Dockerfile`        | Builds a CPU-based runtime with all dependencies |
| `docker_build.sh`   | Builds the Docker image                          |
| `docker_bash.sh`    | Opens an interactive shell inside the container  |
| `docker_jupyter.sh` | Runs Jupyter Notebook inside the container       |

---

#### Build the Docker Image

From the project root:

```bash
./docker_build.sh
```

This creates a Docker image with:

* Python environment
* Required ML libraries (TensorFlow, PyTorch, ONNX Runtime)
* Project source code

---

#### Run Jupyter Notebook in Docker

To start Jupyter inside the container:

```bash
./docker_jupyter.sh
```

This:

* Launches Jupyter on port `8888`
* Mounts the project directory into the container
* Allows notebooks to be executed without local installs

Access Jupyter at:

```
http://localhost:8888
```

---

#### Run an Interactive Shell (Debugging)

For debugging or manual commands:

```bash
./docker_bash.sh
```

This drops you into a shell inside the container with the project environment ready.

---

#### Running the FastAPI Server in Docker

Once inside the container:

```bash
uvicorn ONNX_Fake_News_Detection_utils:create_fastapi_app --factory --host 0.0.0.0 --port 8000
```

The API will be accessible at:

```
http://localhost:8000/docs
```

---

#### Notes and Assumptions

* The provided Docker setup is **CPU-only**
* GPU acceleration is intentionally disabled for portability
* ONNX Runtime provides efficient CPU inference

---

#### When to Use Docker vs Local Setup

| Scenario                   | Recommended      |
| -------------------------- | ---------------- |
| Running notebooks locally  | Local Python env |
| Reproducible grading       | Docker           |
| API deployment demo        | Docker           |
| Avoiding dependency issues | Docker           |


---

### Miscellaneous

* **`requirements.txt`**
  Lists all Python dependencies required to run the project.



## Setup Instructions

### 1. Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:

* TensorFlow
* PyTorch
* Transformers
* ONNX
* ONNX Runtime
* FastAPI
* scikit-learn

---

### 3. Prepare Dataset

Place the Kaggle dataset files in:

```
data/Fake.csv
data/True.csv
```

---

## Typical Workflow

1. Train LSTM model
2. Convert trained model to ONNX
3. Validate inference correctness
4. Compare runtime performance
5. Serve predictions using FastAPI

All steps are demonstrated via notebooks using the shared API.

---

## Notes on DistilBERT

DistilBERT is included as a **bonus exploration**:

* Demonstrates ONNX export for transformer models
* Highlights differences in conversion complexity and runtime behavior
* Not used in the API layer

This separation keeps the API stable and simple.

---

## Summary

This project demonstrates how to:

* Build a text classification system
* Transition from training frameworks to ONNX
* Validate correctness and speed
* Design a clean internal API around ML models

The emphasis is on **deployment-ready ML**, not just model accuracy.

**NOTE** : models and data are added to `.gitignore` as they are too large to push into github. Data can be downloaded from [kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and models can be obtained by running `ONNX_Fake_News_Detection.example.ipynb`