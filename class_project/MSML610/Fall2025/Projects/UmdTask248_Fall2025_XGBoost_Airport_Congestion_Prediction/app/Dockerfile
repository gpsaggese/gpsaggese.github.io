# app/Dockerfile

# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app.py .

# Create data folder (we will mount it later)
RUN mkdir -p /app/data

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]