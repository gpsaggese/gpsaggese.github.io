# Frontend Dockerfile for Fake News Detector
# Builds both Flask backend and Streamlit frontend in one container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY frontend/ ./frontend/
COPY Code_files/ ./Code_files/

# Expose ports
# 5001 for Flask backend
# 8501 for Streamlit frontend
EXPOSE 5001 8501

# Set working directory to frontend
WORKDIR /app/frontend

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5001/api/health || exit 1

# Start both services
# In production, use a process manager like supervisord
# For now, start backend in background and frontend in foreground
CMD python backend.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0
