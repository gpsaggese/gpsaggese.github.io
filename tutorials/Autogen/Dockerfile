# Use Python 3.12 slim (already has Python and pip)
FROM python:3.12-slim

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates (needed for HTTPS) and basic build tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Upgrade pip, setuptools, wheel, and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["bash"]
