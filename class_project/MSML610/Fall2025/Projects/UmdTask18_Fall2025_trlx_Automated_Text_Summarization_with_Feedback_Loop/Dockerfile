# RLHF News Summarization System - Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter notebook
RUN pip install --no-cache-dir jupyter notebook

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy project files
COPY . .

# Expose ports for all services
EXPOSE 8888 8000 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Default command: Start all services
CMD bash -c "\
    echo '================================================================='; \
    echo 'RLHF News Summarization System'; \
    echo '================================================================='; \
    echo ''; \
    echo 'Starting all services...'; \
    echo ''; \
    \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token='' --ServerApp.password='' \
    --IdentityProvider.token='' --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_origin='*' > /tmp/jupyter.log 2>&1 & \
    sleep 2; \
    echo ' Jupyter Notebook started'; \
    \
    python web/backend.py > /tmp/backend.log 2>&1 & \
    sleep 2; \
    echo ' Backend API started'; \
    \
    cd web && python -m http.server 8080 > /tmp/frontend.log 2>&1 & \
    sleep 2; \
    echo ' Web Interface started'; \
    \
    echo ''; \
    echo '================================================================='; \
    echo 'All services running!'; \
    echo '================================================================='; \
    echo ''; \
    echo 'CLICK THESE LINKS TO OPEN IN BROWSER:'; \
    echo ''; \
    echo '  Web Interface (Test Summarizer):'; \
    echo '     http://localhost:8080'; \
    echo ''; \
    echo '  Jupyter Notebooks:'; \
    echo '     http://localhost:8888'; \
    echo ''; \
    echo '  Backend API Docs:'; \
    echo '     http://localhost:8000/docs'; \
    echo ''; \
    echo '================================================================='; \
    echo 'Press Ctrl+C to stop all services'; \
    echo '================================================================='; \
    echo ''; \
    \
    tail -f /tmp/jupyter.log /tmp/backend.log /tmp/frontend.log"
