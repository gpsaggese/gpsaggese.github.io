#!/usr/bin/env bash
set -e

# Activate virtual environment
source /venv/bin/activate

# Move to project workspace
cd /workspace

# Start Jupyter Notebook (old UI)
jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --allow-root \
    --no-browser