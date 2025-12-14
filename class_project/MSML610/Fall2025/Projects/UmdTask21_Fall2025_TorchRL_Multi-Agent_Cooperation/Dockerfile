# TorchRL_MAC container for multi-agent cooperation demo
FROM python:3.10-slim

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    jupyterlab \
    matplotlib

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
