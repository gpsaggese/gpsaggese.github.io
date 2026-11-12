FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
# This tutorial does not need JAX to reserve most of the GPU up front.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /git_root

RUN mkdir -p /install
COPY requirements.txt /install/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        jupyterlab \
        jupyterlab_vim \
        jupytext==1.19.3 \
        -r /install/requirements.txt
        
RUN python -c "import jax, mctx; print('jax', jax.__version__); print('mctx', mctx.__version__)"

COPY run_jupyter.sh /install/run_jupyter.sh
COPY version.sh /install/version.sh
RUN chmod +x /install/run_jupyter.sh /install/version.sh
RUN JAX_PLATFORMS=cpu /install/version.sh 2>&1 | tee /install/version.log

EXPOSE 8888
