#!/bin/bash
echo "Starting Jupyter at http://localhost:8888"
cd "$(dirname "$0")/.."
docker run --rm -it \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -w /workspace \
    electricity-forecasting:latest \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
