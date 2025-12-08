# RLHF News Summarization System

A production-ready news summarization system using Direct Preference Optimization (DPO) to align T5-large with human preferences. The system provides intelligent text summarization with support for multiple input formats, hierarchical processing, and iterative refinement through a modern web interface.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Web Interface](#web-interface)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

This project implements a complete Reinforcement Learning from Human Feedback (RLHF) pipeline for news summarization. The system trains a T5-large model using:

1. **Supervised Fine-Tuning (SFT)** on CNN/DailyMail dataset
2. **LoRA** (Low-Rank Adaptation) for parameter-efficient training
3. **DPO** (Direct Preference Optimization) for human alignment

The final model achieves **72% preference accuracy** and provides a modular, production-ready API with web interface for real-world deployment.

### Why This Project?

- **Human-Aligned Summaries**: DPO training ensures outputs match human preferences
- **Production Ready**: Clean API, error handling, comprehensive logging
- **Multi-Format Support**: Text, URLs, PDF, DOCX, TXT files
- **Intelligent Processing**: Hierarchical summarization for long documents
- **User Refinement**: Iterative improvement with natural language feedback
- **Modern Web UI**: FastAPI backend + responsive frontend

## Key Features

### Core Capabilities

- **Multi-Format Input**: Direct text, URLs (with trafilatura), PDF/DOCX/TXT files
- **Intelligent Chunking**: Sentence-aware splitting with configurable token limits
- **Hierarchical Summarization**: Recursive aggregation for documents >900 tokens
- **Instruction Following**: Natural language commands (e.g., "make it brief", "3 detailed paragraphs", "500 words")
- **Iterative Refinement**: Improve summaries with feedback without retraining
- **Batch Processing**: Multiple URLs or files with combined/separate outputs

### Technical Highlights

- **DPO-Optimized Model**: T5-large trained with Direct Preference Optimization
- **Parameter Efficiency**: LoRA reduces trainable parameters by 99.7%
- **Device Flexibility**: MPS (Apple Silicon), CUDA, or CPU with automatic fallback
- **Modular Architecture**: 12 Python modules with clean separation of concerns
- **Comprehensive Testing**: Example notebook with all functions demonstrated

## System Architecture

```
RLHF_News_Summarization_System/
|
|-- data/                          # Datasets and trained models
|   |-- models/                    # Model checkpoints
|   |   |-- t5-small/              # SFT baseline (T5-small)
|   |   |-- t5-large/              # LoRA T5-large checkpoints
|   |   |-- BART-large/            # LoRA BART (comparison)
|   |   `-- RLHF-t5-large-merged-dpo/  # Final DPO model
|   |-- processed/                 # Preprocessed CNN/DailyMail
|   `-- rlhf/                      # RLHF training data
|
|-- scripts/                       # Modular summarization pipeline
|   |-- input_processing/          # URL, PDF, DOCX extractors
|   |-- summarization/             # Model loader, chunker, summarizer
|   |-- refinement/                # Instruction parser, prompt builder
|   |-- pipeline/                  # End-to-end orchestration
|   `-- examples/                  # Example usage scripts
|
|-- notebooks/                     # Jupyter notebooks
|   |-- RLHF_News_Summarization_System.Example.ipynb  # Main demo
|   |-- data_preparation_and_baseline_t5.ipynb        # Stage 1-2
|   |-- lora_comparison.ipynb                         # Stage 3
|   |-- RLHF_DPO.ipynb                                # Stage 4-5
|   `-- trlx.API.ipynb                                # TRLX tutorial
|
|
|-- web/                           # Web interface
|   |-- backend.py                 # FastAPI server
|   |-- index.html                 # Frontend UI
|   |-- script.js                  # Client logic
|   |-- styles.css                 # Styling
|   `-- src/                       # Additional web assets
|
|-- trlx_custom/                   # Custom TRLX components
|   |-- pipeline/                  # Custom DPO pipeline
|   `-- trainer/                   # Custom DPO trainer
|
|-- utils.py                       # Main API (high-level functions)
|-- requirements.txt               # Python dependencies
|-- docker_build.sh                # Docker build script
|-- docker_bash.sh                 # Docker bash access
|-- docker_jupyter.sh              # Docker Jupyter launcher
|-- README.md                      # This file
|-- RLHF_News_Summarization_System.Example.md  # Notebook documentation
`-- trlx.API.md                    # TRLX API documentation
```


### Pipeline Flow

```
Input (Text/URL/File)
    |
    v
[Input Processing] --> Extract & Clean Text
    |
    v
[Chunking] --> Sentence-aware splitting (~900 tokens)
    |
    v
[Summarization] --> DPO T5-large generates summaries
    |
    v
[Aggregation] --> Hierarchical combination (if >1 chunk)
    |
    v
Output (Summary + Metadata)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- 16GB+ RAM recommended
- GPU/MPS optional (3-5x faster than CPU)
- 10GB+ disk space for models

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RLHF_News_Summarization_System
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (for sentence tokenization):
```python
python -c "import nltk; nltk.download('punkt')"
```

### Optional Dependencies

For full functionality:
```bash
pip install trafilatura      # URL extraction
pip install pdfplumber        # PDF support
pip install python-docx       # DOCX support
```

## Quick Start

### Python API

```python
from utils import summarize_text, summarize_url, summarize_file

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

### Web Interface

1. **Start the backend**:
```bash
python web/backend.py
```

2. **Open frontend**:
```bash
# In a new terminal
cd web
python -m http.server 8080
```

3. **Access**: Navigate to `http://localhost:8080` in your browser

## Usage Examples

### Basic Summarization

```python
from utils import summarize_text

article = """
Artificial intelligence has made remarkable progress in recent years.
Machine learning algorithms can now perform tasks that were once thought
to require human intelligence. Deep learning has been particularly successful
in areas like computer vision and natural language processing.
"""

result = summarize_text(article)
print(result["summary"])
# Output: "AI has advanced significantly, with ML algorithms performing complex tasks..."
```

### With Instructions

```python
# Brief summary
result = summarize_text(article, instructions="make it brief, one sentence")

# Detailed paragraphs
result = summarize_text(article, instructions="3 detailed paragraphs")

# Specific word count
result = summarize_text(article, instructions="500 words")

# Bullet points
result = summarize_text(article, instructions="format as bullet points")
```

### Multiple URLs

```python
from utils import summarize_urls

urls = [
    "https://site1.com/article1",
    "https://site2.com/article2",
    "https://site3.com/article3"
]

# Combined summary
result = summarize_urls(urls, instructions="compare key themes", combine=True)
print(result["summary"])

# Separate summaries
result = summarize_urls(urls, combine=False)
for i, summary in enumerate(result["summaries"]):
    print(f"Article {i+1}: {summary}")
```

### Iterative Refinement

```python
from utils import summarize_text, refine_summary

# Initial summary
result = summarize_text(article)
original = result["summary"]

# Refine with feedback
refined = refine_summary(original, "make it shorter and use bullet points")
print(refined["refined_summary"])

# Further refinement
refined2 = refine_summary(refined["refined_summary"], "add more specific details")
print(refined2["refined_summary"])
```

### Batch File Processing

```python
from utils import summarize_files

files = ["report1.pdf", "report2.pdf", "notes.txt"]

# Combined summary of all files
result = summarize_files(files, instructions="executive summary", combine=True)
print(result["summary"])
```

## Project Structure

### Core Modules

#### `scripts/input_processing/`
- **url_extractor.py**: Extract article text from URLs using trafilatura
- **pdf_extractor.py**: Extract text from PDFs using pdfplumber
- **docx_extractor.py**: Extract text from DOCX files using python-docx
- **text_cleaner.py**: Normalize whitespace, remove special characters

#### `scripts/summarization/`
- **model_loader.py**: Load DPO T5-large with device auto-detection
- **chunker.py**: Sentence-aware text splitting with overlap
- **summarizer.py**: Generate summaries with instruction parsing
- **aggregator.py**: Hierarchical combination of chunk summaries

#### `scripts/refinement/`
- **instruction_parser.py**: Parse natural language instructions
- **prompt_builder.py**: Build T5-compatible prompts

#### `scripts/pipeline/`
- **summarization_pipeline.py**: End-to-end orchestration

### Web Interface

- **backend.py**: FastAPI server with CORS, endpoints for all functions
- **index.html**: Responsive UI with tabbed interface (Text/URL/File)
- **script.js**: Client-side logic, API calls, message rendering
- **style.css**: Modern styling with glassmorphism effects

## Training Pipeline

### Stage 1: Data Preparation

```bash
# Covered in: notebooks/data_preparation_and_baseline_t5.ipynb
```

- Load CNN/DailyMail from Hugging Face (287k training examples)
- Tokenize with T5 tokenizer
- Create train/val/test splits
- Save processed dataset

### Stage 2: Supervised Fine-Tuning (SFT)

```bash
# Train T5-small baseline (in data_preparation_and_baseline_t5.ipynb)
```

- Full fine-tuning on CNN/DailyMail
- 50 examples, 3 epochs (demo)
- ROUGE-L: ~0.29
- Checkpoint: `data/models/t5-small/`

### Stage 3: LoRA Fine-Tuning

```bash
# Covered in: notebooks/lora_comparison.ipynb
```

- Apply LoRA to T5-large (rank=8, alpha=32)
- Train only 0.3% of parameters
- Compare with BART-large
- ROUGE-L: 0.32 (T5) vs 0.31 (BART)
- Merge LoRA weights: `data/models/t5-large/`

### Stage 4: Preference Pair Generation

```python
# Generate DPO training data (in notebooks/RLHF_DPO.ipynb)
```

- T5-large summaries = "chosen" (higher quality)
- T5-small summaries = "rejected" (lower quality)
- 400 preference pairs
- Save as JSONL: `data/rlhf/dpo_pairs.jsonl`

### Stage 5: DPO Training

```bash
# Covered in: notebooks/RLHF_DPO.ipynb
```

- TRLX library with custom DPO trainer (trlx_custom/)
- Optimize for human-aligned outputs
- Final model: `data/models/RLHF-t5-large-merged-dpo/`
- **72% preference accuracy**

## Web Interface

### Features

- **Three Input Modes**: Text, URL, File upload
- **Instructions Field**: Natural language commands
- **Real-time Processing**: Loading indicators, error handling
- **Metadata Display**: Chunks, input/output lengths, processing time
- **Responsive Design**: Works on desktop and mobile
- **Multiple File Support**: Upload and process multiple files at once

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/summarize/text` | POST | Summarize text |
| `/summarize/url` | POST | Summarize from URL |
| `/summarize/urls` | POST | Summarize multiple URLs |
| `/summarize/file` | POST | Summarize uploaded file |
| `/summarize/files` | POST | Summarize multiple files |
| `/refine` | POST | Refine existing summary |

### Starting the Web Interface

```bash
# Terminal 1: Backend
python web/backend.py
# Server runs on http://localhost:8000

# Terminal 2: Frontend
cd web && python -m http.server 8080
# UI available at http://localhost:8080
```

## API Reference

### `summarize_text(text, instructions=None, clean_text=True)`

Summarize raw text.

**Parameters:**
- `text` (str): Input text to summarize
- `instructions` (str, optional): User instructions (e.g., "brief", "500 words")
- `clean_text` (bool): Whether to clean text first (default: True)

**Returns:** Dictionary with:
- `summary` (str): Generated summary
- `num_chunks` (int): Number of chunks processed
- `input_length` (int): Input character count
- `summary_length` (int): Output character count

### `summarize_url(url, instructions=None)`

Summarize article from URL.

**Parameters:**
- `url` (str): URL to summarize
- `instructions` (str, optional): User instructions

**Returns:** Dictionary with summary + metadata (title, author, date)

### `summarize_urls(urls, instructions=None, combine=True)`

Summarize multiple URLs.

**Parameters:**
- `urls` (List[str]): List of URLs
- `instructions` (str, optional): User instructions
- `combine` (bool): Combine into single summary or separate (default: True)

**Returns:** Dictionary with combined or individual summaries

### `summarize_file(filepath, instructions=None)`

Summarize from file (PDF, DOCX, TXT).

**Parameters:**
- `filepath` (str): Path to file
- `instructions` (str, optional): User instructions

**Returns:** Dictionary with summary + file metadata

### `summarize_files(filepaths, instructions=None, combine=True)`

Summarize multiple files.

**Parameters:**
- `filepaths` (List[str]): List of file paths
- `instructions` (str, optional): User instructions
- `combine` (bool): Combine into single summary (default: True)

**Returns:** Dictionary with combined or individual summaries

### `refine_summary(summary, feedback, original_text=None)`

Refine existing summary based on feedback.

**Parameters:**
- `summary` (str): Original summary
- `feedback` (str): User feedback (e.g., "make it shorter")
- `original_text` (str, optional): Original text for context

**Returns:** Dictionary with refined summary

### `get_model_info()`

Get model information.

**Returns:** Dictionary with model type, name, device, max_length, vocab_size

## Performance

### Model Comparison

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable Params |
|-------|---------|---------|---------|------------------|
| T5-small (SFT) | 0.41 | 0.19 | 0.29 | 60M (100%) |
| T5-large (LoRA) | 0.45 | 0.22 | 0.32 | 2M (0.3%) |
| BART-large (LoRA) | 0.44 | 0.21 | 0.31 | 2M (0.3%) |
| **T5-large (DPO)** | **0.46** | **0.23** | **0.33** | **2M (0.3%)** |

### Speed Benchmarks

| Device | Summarization Time (1000 tokens) |
|--------|----------------------------------|
| MPS (Apple M1) | ~2-3 seconds |
| CUDA (RTX 3090) | ~1-2 seconds |
| CPU (Intel i7) | ~8-12 seconds |

### Key Findings

1. **DPO Improvement**: +3% ROUGE-L over LoRA baseline
2. **T5 vs BART**: T5-large outperforms BART-large across all metrics
3. **Parameter Efficiency**: LoRA achieves 99.7% parameter reduction with minimal performance loss
4. **Preference Accuracy**: 72% on held-out preference pairs

## Development

### Running Tests

```bash
# Test individual components
python scripts/summarization/model_loader.py
python scripts/summarization/chunker.py
python scripts/pipeline/summarization_pipeline.py

# Test utils API
python utils.py
```

### Example Notebook

The complete example notebook demonstrates all functionality:

```bash
jupyter notebook notebooks/RLHF_News_Summarization_System.Example.ipynb
```

**Sections:**
1. Data preparation
2. SFT training (T5-small)
3. LoRA training (T5-large, BART-large)
4. Preference pair generation
5. DPO training
6. Evaluation and comparison
7. **Modular pipeline demo** (Stage 7) - Can run independently!

**Additional Notebooks:**
- `notebooks/data_preparation_and_baseline_t5.ipynb` - Detailed data prep and SFT
- `notebooks/lora_comparison.ipynb` - LoRA implementation and T5 vs BART
- `notebooks/RLHF_DPO.ipynb` - Complete DPO training workflow
- `notebooks/trlx.API.ipynb` - TRLX library tutorial

### Code Style

- **Modular**: Each component has single responsibility
- **Documented**: Comprehensive docstrings
- **Typed**: Type hints for all functions
- **Logged**: Informative logging throughout
- **Error Handling**: Graceful degradation

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure you're in project root
cd /path/to/RLHF_News_Summarization_System
python -c "from utils import summarize_text; print('Success!')"
```

#### Model Not Found

```bash
# Check model exists
ls data/models/RLHF-t5-large-merged-dpo/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

#### Out of Memory

- Reduce chunk size in `scripts/summarization/chunker.py`
- Use CPU instead of GPU: Set `device="cpu"` in model_loader.py
- Process smaller batches

#### URL Extraction Fails

- Install trafilatura: `pip install trafilatura`
- Some websites block automated access
- Try different URLs or use PDF/text export

#### Slow Performance

- Use GPU/MPS if available
- Check device: `python -c "from utils import get_model_info; print(get_model_info())"`
- Ensure MPS fallback is enabled (set in backend.py)

### Word Count Issues

If requested word count doesn't match output:
- System uses token-based generation (1 word ≈ 1.3 tokens)
- T5 may stop early if input is too short
- Use "detailed" or "comprehensive" for longer outputs
- Avoid requesting more words than input contains

## References

### Papers

- **T5**: Raffel et al. (2020) - [Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- **DPO**: Rafailov et al. (2023) - [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **LoRA**: Hu et al. (2021) - [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **ROUGE**: Lin (2004) - Automatic Evaluation of Summaries

### Datasets

- **CNN/DailyMail**: Hermann et al. (2015) - [Hugging Face](https://huggingface.co/datasets/abisee/cnn_dailymail)

### Libraries

- **Transformers**: [Hugging Face](https://github.com/huggingface/transformers)
- **PEFT**: [Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- **TRLX**: [CarperAI](https://github.com/CarperAI/trlx)
- **Trafilatura**: [Web Scraping](https://github.com/adbar/trafilatura)

## License

This project is part of the RLHF News Summarization System coursework.

## Acknowledgments

- CNN/DailyMail dataset creators
- Hugging Face for Transformers library
- CarperAI for TRLX
- T5, DPO, and LoRA paper authors

---

**For detailed implementation guides, see:**
- [scripts/README.md](scripts/README.md) - Modular pipeline documentation
- [RLHF_News_Summarization_System.example.md](RLHF_News_Summarization_System.example.md) - Notebook documentation
