# =============================================================================
# Dockerfile for BoTorch Multi-Objective Drug Discovery Pipeline
# =============================================================================
# This image provides a complete environment for running Bayesian optimization
# workflows for drug discovery using BoTorch, GPyTorch, and RDKit.
# =============================================================================

FROM python:3.11-slim

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------
LABEL maintainer="Drug Discovery Team"
LABEL description="Multi-Objective Bayesian Optimization for Drug Discovery"
LABEL version="1.0"

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Create Application Directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python Dependencies
# -----------------------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Copy Application Code
# -----------------------------------------------------------------------------
COPY Botorch_utils.py .
COPY Botorch_example.ipynb .
COPY Botorch_API.ipynb .

# -----------------------------------------------------------------------------
# Create directories for data and results
# -----------------------------------------------------------------------------
RUN mkdir -p /app/data /app/results

# -----------------------------------------------------------------------------
# Expose Jupyter Port
# -----------------------------------------------------------------------------
EXPOSE 8888

# -----------------------------------------------------------------------------
# Default Command - Start Jupyter Lab
# -----------------------------------------------------------------------------
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
