FROM python:3.10-slim

# OS deps (include libgomp1 to avoid the scikit-learn TLS/libgomp issue on arm64)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Core Jupyter + libs (avoid the deprecated "jupyter" meta package)
RUN pip install --no-cache-dir \
      notebook==6.5.7 jupyterlab==4.2.5 \
      nbconvert==7.16.4 nbclient==0.10.0 jupyter-core==5.7.2 && \
    pip install --no-cache-dir \
      torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      scikit-learn==1.3.2 transformers==4.44.2 sentence-transformers==3.1.1 \
      umap-learn matplotlib pyarrow==12.0.1 tqdm

# Bring project in
COPY . /workspace

EXPOSE 8888
ENV TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

CMD ["bash","-lc","jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token= --notebook-dir=/workspace"]
