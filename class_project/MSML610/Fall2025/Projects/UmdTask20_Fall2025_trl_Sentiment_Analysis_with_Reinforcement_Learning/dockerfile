# Use the exact Python version you are running locally
FROM python:3.11.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# 'git' is required for installing packages directly from GitHub or Hugging Face
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port for Jupyter
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]