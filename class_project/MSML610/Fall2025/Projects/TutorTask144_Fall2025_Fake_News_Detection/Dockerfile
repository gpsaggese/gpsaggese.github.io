# Multi-stage Dockerfile for Fake News Detection Project
# Stage 1: Base image with Python and dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab averaged_perceptron_tagger wordnet omw-1.4 && \
    python -m spacy download en_core_web_sm 2>/dev/null || true

# Stage 3: Final application image
FROM dependencies as application

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/data \
    && mkdir -p /app/models \
    && mkdir -p /app/output \
    && mkdir -p /app/logs

# Make startup script executable
RUN chmod +x /app/docker-entrypoint.sh || true

# Expose ports
EXPOSE 8000 5000

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["all"]
