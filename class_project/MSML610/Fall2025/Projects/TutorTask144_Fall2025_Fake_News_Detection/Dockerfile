# BERT Fake News Detection - Docker Container
# Lightweight Python-based image optimized for deep learning

FROM python:3.9-slim

# Set metadata
LABEL maintainer="BERT Fake News Detection Project"
LABEL description="BERT-based fake news detection system with support for multiple datasets"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy project source code
COPY bert_utils.py .
COPY train_bert_liar_only.py .
COPY train_bert_model.py .
COPY eval_bert_liar.py .
COPY BERT.API.md .
COPY BERT.example.md .
COPY BERT.API.ipynb .
COPY BERT.example.ipynb .

# Create necessary directories
RUN mkdir -p /app/data \
    && mkdir -p /app/models \
    && mkdir -p /app/logs

# Create volume mount points for data and models
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Expose Jupyter port (8888) and API port (5000)
EXPOSE 8888 5000

# Default command: Start Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# Alternative commands:
# For training BERT on LIAR: docker run <image> python train_bert_liar_only.py
# For training on multiple datasets: docker run <image> python train_bert_model.py
# For evaluation: docker run <image> python eval_bert_liar.py
# For interactive Python: docker run -it <image> python
# For bash shell: docker run -it <image> /bin/bash
