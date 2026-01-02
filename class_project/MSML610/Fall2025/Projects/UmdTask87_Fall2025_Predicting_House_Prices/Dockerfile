FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Kaggle CLI credentials mount point
ENV KAGGLE_CONFIG_DIR=/root/.kaggle

# Project files
COPY . .

EXPOSE 8888 8000
CMD ["bash"]

