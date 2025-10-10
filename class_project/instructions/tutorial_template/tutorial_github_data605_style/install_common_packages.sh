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

echo ">>> Installing Python packages..."
pip3 install --no-cache-dir \
    ipython \
    tornado==6.1 \
    jupyter-client==7.3.2 \
    jupyter-contrib-core \
    jupyter-contrib-nbextensions \
    yapf \
    psycopg2-binary \
    numpy \
    pandas \
    matplotlib \
    seaborn 



