# Dockerfile for UmdTask88 - DGL Fraud Detection in Credit Card Transactions
# CPU-only base to keep image size lean; install PyTorch via pip
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Optional: extra system deps for plotting / I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps for this project
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Project code
COPY . .

# Make sure these exist in the container
RUN mkdir -p data/raw data/processed data/artifacts notebooks

EXPOSE 8888

CMD jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
