FROM --platform=$BUILDPLATFORM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential \
      libxml2-dev libxslt-dev zlib1g-dev libpq-dev \
      git curl vim && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Core notebook + scientific stack
RUN pip3 install --no-cache-dir \
    ipython jupyter jupyter-client==7.3.2 tornado==6.1 \
    numpy pandas matplotlib seaborn pillow scikit-learn tqdm

# JAX/Flax/Optax (CUDA 12.x)
RUN pip3 install --no-cache-dir \
    --upgrade "jax[cuda12_pip]==0.4.28" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip3 install --no-cache-dir flax optax

EXPOSE 8888
