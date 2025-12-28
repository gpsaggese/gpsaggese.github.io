# Base Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /workspace

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ../requirements.txt /workspace/requirements.txt

# Create a virtual environment
RUN python3 -m venv /venv

# Install requirements
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install -r requirements.txt

# Copy full project into container
COPY .. /workspace

# Expose Jupyter port
EXPOSE 8888

# Default command runs your Jupyter server
CMD ["/workspace/docker/run_jupyter_server.sh"]