#!/bin/bash
# Start Jupyter Notebook inside the MSML610 Docker container
# Notebooks root: the mounted project directory at /curr_dir

set -e

PROJECT_DIR="/curr_dir"
PORT="${JUPYTER_HOST_PORT:-8888}"

cd "$PROJECT_DIR"

echo "Starting Jupyter Notebook in $PROJECT_DIR on port $PORT ..."
echo "You should be able to open it at:  http://localhost:$PORT"

jupyter notebook \
  --ip=0.0.0.0 \
  --port="$PORT" \
  --no-browser \
  --notebook-dir="$PROJECT_DIR" \
  --allow-root
