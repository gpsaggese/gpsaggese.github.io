# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ../data ./data
COPY ../utils ./utils
COPY ../notebooks ./notebooks
COPY ../scripts ./scripts
COPY ../pipelines ./pipelines
COPY ../docs ./docs
COPY ../CLAUDE.md ./CLAUDE.md

# Create directories for outputs
RUN mkdir -p models pipeline_outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port for Jupyter notebook
EXPOSE 8888

# Expose port for TensorFlow Serving
EXPOSE 8501

# Default command: run bash for interactive use
CMD ["/bin/bash"]
