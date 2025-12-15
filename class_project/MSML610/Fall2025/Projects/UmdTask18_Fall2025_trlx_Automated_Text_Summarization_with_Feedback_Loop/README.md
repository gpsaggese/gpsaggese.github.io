# RLHF News Summarization System using trlX (Difficulty Level: Hard)

A production-ready news summarization system using Direct Preference Optimization (DPO) to align T5-large with human preferences. The system provides intelligent text summarization with support for multiple input formats, hierarchical processing, and iterative refinement through a modern web interface.

## Table of Contents
1. [Author](#author)
2. [Overview](#overview)
3. [Project Structure](#project-structure)
4. [How to Run the Project](#how-to-run-the-project)
5. [Python Dependencies](#python-dependencies)
6. [Documentation Links](#documentation-links)
7. [Final Notes and Disclaimers](#final-notes-and-disclaimers)

---

## Authors
**Anvesh Chitturi**, **Sai Dhruv Yellanki Hanmanthrao** 

- Email: achittu1@umd.edu, dhruvsai@umd.edu
- UID's: 121332790, 120998607
---

## Overview

This project implements a complete Reinforcement Learning from Human Feedback (RLHF) pipeline for news summarization. The system demonstrates how to build an explainable, production-ready summarization system using state-of-the-art techniques.

The project includes both:
- **Training notebooks** explaining the complete RLHF pipeline from data preparation to DPO optimization
- **Production system** with modular Python scripts and web interface for real-world deployment

### What This System Does

- **Summarizes text** from multiple sources (direct text, URLs, PDF/DOCX files)
- **Generates human-aligned summaries** using DPO-optimized T5-large model
- **Handles long documents** with intelligent chunking and hierarchical aggregation
- **Follows instructions** like "make it brief" or "detailed summary"
- **Provides web interface** for easy interaction
- **Achieves 72% preference accuracy** on human preference alignment

### Training Pipeline

1. **Supervised Fine-Tuning (SFT)** on CNN/DailyMail dataset
2. **LoRA** (Low-Rank Adaptation) for parameter-efficient training (99.7% parameter reduction)
3. **DPO** (Direct Preference Optimization) for human alignment
4. **Final model**: ROUGE-L score of 0.33 (33%), competitive with state-of-the-art

---

## Project Structure

| File / Folder | Description |
|---------------|-------------|
| `scripts/` | Modular Python scripts for the production summarization pipeline |
| `scripts/input_processing/` | URL, PDF, DOCX extractors and text cleaning |
| `scripts/summarization/` | Model loader, chunker, summarizer, aggregator |
| `scripts/refinement/` | Instruction parser and prompt builder |
| `scripts/pipeline/` | End-to-end pipeline orchestration |
| `scripts/utils.py` | High-level API wrapper functions |
| `notebooks/` | Jupyter notebooks demonstrating the complete training pipeline |
| `notebooks/RLHF_News_Summarization_System.Example.ipynb` | Main demo notebook |
| `notebooks/data_preparation_and_baseline_t5.ipynb` | Stage 1-2: Data prep and SFT |
| `notebooks/lora_comparison.ipynb` | Stage 3: LoRA training |
| `notebooks/RLHF_DPO.ipynb` | Stage 4-5: DPO training |
| `notebooks/trlx.API.ipynb` | TRLX library tutorial |
| `web/` | Web interface (FastAPI backend + responsive frontend) |
| `web/backend.py` | FastAPI server with REST API endpoints |
| `web/index.html` | Frontend UI with tabbed interface |
| `web/script.js` | Client-side JavaScript |
| `web/styles.css` | Modern CSS styling |
| `data/` | Datasets and trained model checkpoints |
| `data/models/RLHF-t5-large-merged-dpo/` | Final DPO-optimized model |
| `data/processed/` | Preprocessed CNN/DailyMail dataset |
| `data/rlhf/` | DPO training preference pairs |
| `trlx_custom/` | Custom TRLX components for DPO training |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker configuration for containerized deployment |
| `README.md` | This file |
| `RLHF_News_Summarization_System.Example.md` | Detailed notebook documentation |

---

## How to Run the Project

We provide multiple ways to run the project: Docker (recommended for consistency) or local Python environment.

### Option 1: Docker (Recommended)

Docker ensures consistent environment setup across all platforms.

#### Build and start all services

```bash
docker-compose up --build
```

This single command will:
- Build the Docker image
- Start all services (Jupyter, Backend API, Web Interface)
- Mount your local `notebooks/` and `data/` directories

#### Access the services

Once running, you can access:
- **Web Interface**: http://localhost:8080
- **Backend API Docs**: http://localhost:8000/docs
- **Jupyter Notebooks**: http://localhost:8888

Press `Ctrl+C` to stop all services.

#### Optional: Run in background

If you want to run services in the background:

```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop background services:
```bash
docker-compose down
```

### Option 2: Local Python Environment

#### Prerequisites

- Python 3.10 or higher
- 16GB+ RAM recommended
- GPU/MPS optional (3-5x faster than CPU)
- 10GB+ disk space for models

#### Step 1: Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Download NLTK data

```python
python -c "import nltk; nltk.download('punkt')"
```

#### Step 4: Start the backend

```bash
python web/backend.py
```

The backend will:
- Load the DPO-optimized T5-large model
- Auto-detect device (MPS/CUDA/CPU)
- Start FastAPI server on http://localhost:8000

#### Step 5: Start the frontend (in a new terminal)

```bash
cd web
python -m http.server 8080
```

Access the web interface at http://localhost:8080

### Option 3: Jupyter Notebooks

To explore the training pipeline:

```bash
jupyter notebook notebooks/RLHF_News_Summarization_System.Example.ipynb
```

**Note**: Models are already trained. The notebook demonstrates the pipeline and can load pre-trained checkpoints.

---

## Python Dependencies

All required Python libraries are listed in `requirements.txt`. Key dependencies include:

### Core Libraries
- `torch==2.1.2` – PyTorch for deep learning
- `transformers==4.44.2` – Hugging Face Transformers
- `datasets==4.4.1` – Dataset loading and processing
- `peft==0.10.0` – Parameter-Efficient Fine-Tuning (LoRA)
- `trlx` – TRLX library for RLHF/DPO training

### Summarization Pipeline
- `trafilatura==1.6.3` – URL text extraction
- `pdfplumber==0.10.3` – PDF text extraction
- `python-docx==1.1.0` – DOCX text extraction
- `nltk==3.8.1` – Sentence tokenization
- `rouge-score==0.1.2` – ROUGE metrics for evaluation

### Web Interface
- `fastapi==0.110.0` – Backend API framework
- `uvicorn==0.29.0` – ASGI server
- `aiofiles==23.2.1` – Async file handling

### Full Installation

```bash
pip install -r requirements.txt
```

**Note**: Some dependencies (like `deepspeed`) are platform-specific and will only install on Linux.

---

## Documentation Links

- [System Architecture Diagram](./RLHF_News_Summarization_System.Example.md#system-architecture) – Visual representation of the complete system
- [Example Notebook Documentation](./RLHF_News_Summarization_System.Example.md) – Comprehensive guide to the training pipeline
- [TRLX API Tutorial](./trlx.API.md) – How to use TRLX for RLHF training

### Quick Start Guides

#### Python API Usage

```python
from scripts.utils import summarize_text, summarize_url, summarize_file

# Summarize text
result = summarize_text("Your long article text here...")
print(result["summary"])

# Summarize from URL
result = summarize_url("https://example.com/article")
print(result["summary"])

# Summarize PDF/DOCX
result = summarize_file("document.pdf", instructions="brief")
print(result["summary"])
```

#### Web Interface

1. Start backend: `python web/backend.py`
2. Start frontend: `cd web && python -m http.server 8080`
3. Open browser: http://localhost:8080
4. Choose input type (Text/URL/File) and click "Generate Summary"

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/summarize/text` | POST | Summarize text |
| `/summarize/url` | POST | Summarize from URL |
| `/summarize/file` | POST | Summarize uploaded file |
| `/refine` | POST | Refine existing summary |

Full API documentation available at: http://localhost:8000/docs

---

## Final Notes and Disclaimers

### Model Configuration

The system uses carefully tuned parameters for high-quality output:

```python
max_length = 400          # ~250-300 words
min_length = 150          # ~6-8 lines minimum
repetition_penalty = 1.5  # Prevents repetitive output
length_penalty = 2.0      # Encourages complete sentences
early_stopping = True     # Stops when quality degrades
```

These parameters ensure grammatically correct, coherent summaries without repetition.



### Model Comparison

| Model | ROUGE-L | Trainable Params |
|-------|---------|------------------|
| T5-small (SFT) | 0.29 | 60M (100%) |
| T5-large (LoRA) | 0.32 | 2M (0.3%) |
| **T5-large (DPO)** | **0.33** | **2M (0.3%)** |

### Key Achievements

- **72% preference accuracy** on human alignment
- **99.7% parameter reduction** with LoRA
- **+3% ROUGE-L improvement** from DPO over LoRA baseline
- **Production-ready** modular system with web interface
- **Multi-format support** (text, URL, PDF, DOCX)
- **Quality output** with anti-repetition mechanisms

### Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce chunk size or use CPU instead of GPU
2. **URL Extraction Fails**: Install `trafilatura` or try different URLs
3. **Slow Performance**: Use GPU/MPS if available
4. **Repetitive Output**: Ensure `repetition_penalty=1.5` in `scripts/summarization/summarizer.py`

**Parameter Changes Not Taking Effect:**

The backend automatically disables Python bytecode caching. If issues persist:
```bash
python web/backend.py
```

### Dataset and Training

- **Dataset**: CNN/DailyMail (287k article-summary pairs)
- **Training Time**: ~7-9 hours for complete pipeline (SFT + LoRA + DPO)
- **Preference Pairs**: 400 pairs (T5-large vs T5-small summaries)

### References

**Papers:**
- T5: Raffel et al. (2020) - [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- DPO: Rafailov et al. (2023) - [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- LoRA: Hu et al. (2021) - [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

**Libraries:**
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [TRLX (CarperAI)](https://github.com/CarperAI/trlx)

### License

This project is part of the MSML 610 course.

### Acknowledgments

- CNN/DailyMail dataset creators
- Hugging Face for Transformers library
- CarperAI for TRLX
- T5, DPO, and LoRA paper authors

---

**For detailed technical documentation, see:**
- [RLHF_News_Summarization_System.Example.md](./RLHF_News_Summarization_System.Example.md) – Complete training pipeline walkthrough
- [trlx.API.md](./trlx.API.md) – TRLX library usage guide
