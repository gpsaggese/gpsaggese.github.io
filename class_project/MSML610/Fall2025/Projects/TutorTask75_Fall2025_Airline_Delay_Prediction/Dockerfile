# Use a public base Python image
FROM python:3.10-slim AS builder

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# Set workdir inside the container
WORKDIR /app

# Copy requirements first, to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src ./src
COPY models ./models
COPY notebooks ./notebooks
COPY reports ./reports
COPY configs ./configs

# Expose the Streamlit port
EXPOSE 8501

# Default command: run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
