FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/work

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /work/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r /work/requirements.txt && \
    python -m pip install jupyterlab notebook ipykernel

COPY . /work

EXPOSE 8888
CMD ["bash"]
