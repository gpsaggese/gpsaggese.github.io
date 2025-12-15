FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Avoid Python writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for building some Python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code (utils, notebooks, data, etc.)
COPY . .

# Expose Jupyter (8888) and FastAPI (8000) ports
EXPOSE 8888
EXPOSE 8000

# Default command: start an interactive shell
# - You can run Jupyter or uvicorn manually from here
CMD ["bash"]
