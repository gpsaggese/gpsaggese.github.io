# Base Python image for Docker
FROM python:3.11-slim

# Set directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (so Docker can cache installations)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the image
COPY . .

# Disable wandb logging inside Docker unless user enables it
ENV WANDB_MODE=disabled

# Default command to run your pipeline
CMD ["python3", "main.py"]

