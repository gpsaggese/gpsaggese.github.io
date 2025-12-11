#!/bin/bash
set -e

echo ">>> Installing common system packages..."
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libpq-dev \
    git \
    vim \
    curl \
    sudo

echo ">>> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

echo ">>> Installing Python packages from pyproject.toml..."
cd /install
uv pip install --system . 



