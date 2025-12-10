# Dockerfile for DeepSpeed Vision Transformer Fine-Tuning
#
# This Dockerfile sets up an environment for training Vision Transformers
# using DeepSpeed ZeRO optimization on multi-GPU systems.
#
# Base image: PyTorch with CUDA support
# Includes: DeepSpeed, Transformers, WandB, and other ML libraries

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    libaio-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install DeepSpeed (if not already in requirements.txt)
# Note: DeepSpeed may need to be built from source for optimal performance
RUN pip install --no-cache-dir deepspeed

# Copy project files
COPY . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden)
CMD ["python", "train_deepspeed_stage3.py", "--help"]

