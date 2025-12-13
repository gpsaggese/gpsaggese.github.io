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
pip3 install --no-cache-dir --break-system-packages \
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

# DAMIAN - Uninstall Scikit-learn and reinstall without OpenMP (It will be much slower but I have an issue with the ARM architecture)
# pip3 uninstall -y scikit-learn --break-system-packages
# export SKLEARN_NO_OPENMP=1
# pip3 install --no-binary scikit-learn scikit-learn --break-system-packages


