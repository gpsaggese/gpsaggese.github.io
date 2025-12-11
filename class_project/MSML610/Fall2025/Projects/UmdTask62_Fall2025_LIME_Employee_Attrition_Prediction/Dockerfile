# Use a specific version of Python
FROM python:3.10-slim

# Install system dependencies that are often needed for machine learning packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy the requirements file first to take advantage of Docker's layer caching
COPY requirements.txt /workspace/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . /workspace

# Expose port 8888 (Jupyter Notebook default port)
EXPOSE 8888

# Set environment variables (to disable some libraries' verbose output)
ENV TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

# Start Jupyter Notebook server
CMD ["bash", "-lc", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token= --notebook-dir=/workspace"]
