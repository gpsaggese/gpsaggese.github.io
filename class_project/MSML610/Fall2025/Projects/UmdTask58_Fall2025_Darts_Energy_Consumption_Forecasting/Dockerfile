# Dockerfile for Energy Consumption Forecasting with Darts
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Builder - Install Python dependencies
# =============================================================================
FROM base as builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy project files
COPY --chown=appuser:appuser . .

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Default command - start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Stage 4: Development image (includes additional dev tools)
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install \
    jupyterlab \
    black \
    flake8 \
    pytest \
    ipywidgets

# Enable Jupyter widgets
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix || true

USER appuser

# Default command for development - start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

