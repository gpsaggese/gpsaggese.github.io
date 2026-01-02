FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy requirements first for caching
COPY requirements.txt .

# Install pip dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . /app

# Make sure Python can import /deploy modules
ENV PYTHONPATH="/app:/app/pipeline"

# HuggingFace cache directory
ENV HF_HOME=/root/.cache/huggingface

# Run the pipeline
CMD ["python", "pipeline/run_pipeline.py"]
