# FairnessPP Docker Configuration
# Course: MSML610 Fall 2025
# Project: Fairness in Predictive Policing

FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY FairnessPP_utils.py .
COPY FairnessPP.API.ipynb .
COPY FairnessPP.API.md .
COPY FairnessPP.example.ipynb .
COPY FairnessPP.example.md .
COPY README.md .

# Create data directory
RUN mkdir -p data

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]