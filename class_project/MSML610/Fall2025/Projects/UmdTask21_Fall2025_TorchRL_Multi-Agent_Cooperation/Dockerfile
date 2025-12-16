FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/tmp \
    SDL_VIDEODRIVER=dummy

# System deps for pygame + basic headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
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

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /workspace/requirements.txt

# Fail-fast sanity import during build
RUN python -c "import torch, tensordict, torchrl; print('torch', torch.__version__, 'tensordict', tensordict.__version__, 'torchrl', torchrl.__version__)"

COPY . /workspace

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.allow_origin=*"]
