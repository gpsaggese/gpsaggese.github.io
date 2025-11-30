FROM python:3.11-slim

# Install system dependencies for LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY .. /app

# Default command prints help; override in docker run command
CMD ["bash", "-c", "echo 'Docker image built. To run scripts, override the CMD when running the container.'"]