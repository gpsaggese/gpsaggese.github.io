# ===============================================
# MSML610 Fall2025 - Weights & Biases Transfer Learning Project
# ===============================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Metadata
LABEL maintainer="pranavv1@umd.edu"
LABEL course="MSML610"
LABEL term="Fall2025"
LABEL project="Weights_and_Biases_Image_Classification_with_Transfer_Learning"

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies directly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        kagglehub \
        IPython \
        pandas \
        numpy \
        tensorflow \
        keras \
        wandb \
        python-dotenv \
        ipywidgets \
        jupyter \
        matplotlib   

# Expose Jupyter Notebook port
EXPOSE 8888

# Default command: start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
