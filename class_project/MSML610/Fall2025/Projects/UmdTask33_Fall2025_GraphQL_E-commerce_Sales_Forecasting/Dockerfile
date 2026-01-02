# Use a small official Python image
FROM python:3.11-slim

# Some sensible Python defaults
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# System deps (needed for numpy / pandas / scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy ALL project files into the image
# (code, models/, raw/ CSVs, notebooks, etc.)
COPY . .

# Expose the API port
EXPOSE 8000

# Default command: start the GraphQL API
CMD ["uvicorn", "GraphQL_API:app", "--host", "0.0.0.0", "--port", "8000"]
