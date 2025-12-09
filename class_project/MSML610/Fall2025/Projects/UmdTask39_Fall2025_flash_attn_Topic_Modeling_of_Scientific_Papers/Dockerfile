FROM python:3.10-slim

# OS deps (include libgomp1 to avoid the scikit-learn TLS/libgomp issue on arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for better layer caching
COPY requirements.txt /workspace/

# Install PyTorch first with the correct index, then all other deps
RUN pip install --no-cache-dir \
      torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# Bring project in
COPY . /workspace

EXPOSE 8888
ENV TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

CMD ["bash","-lc","jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token= --notebook-dir=/workspace"]
