# Build stage: install Python deps that require C compilation (pygame, numpy,
# matplotlib). Keeps build-essential out of the final image.
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# SDL2-dev packages pull in their runtime counterparts as transitive deps,
# so pygame compiles and links correctly here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --prefix=/install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Runtime stage: no compiler, no git, no curl — only what the app needs.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/tmp \
    SDL_VIDEODRIVER=dummy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages and entry-point scripts.
COPY --from=builder /install /usr/local

WORKDIR /workspace

# Fail-fast sanity import during build.
RUN python -c "import torch, tensordict, torchrl; print('torch', torch.__version__, 'tensordict', tensordict.__version__, 'torchrl', torchrl.__version__)"

COPY . /workspace

EXPOSE 8888
