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
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn

echo ">>> Installing Google Cloud packages..."
pip3 install --no-cache-dir \
    google-cloud-storage \
    google-cloud-aiplatform \
    google-auth \
    google-cloud-core \
    cloudml-hypertune

echo ">>> Installing NLP packages..."
pip3 install --no-cache-dir \
    nltk \
    spacy \
    textblob \
    contractions \
    emoji

echo ">>> Installing ML/Transformers packages..."
pip3 install --no-cache-dir \
    torch \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    protobuf

echo ">>> Installing additional packages..."
pip3 install --no-cache-dir \
    tqdm \
    python-dotenv \
    pytz \
    plotly \
    jsonschema

echo ">>> All packages installed successfully!"
