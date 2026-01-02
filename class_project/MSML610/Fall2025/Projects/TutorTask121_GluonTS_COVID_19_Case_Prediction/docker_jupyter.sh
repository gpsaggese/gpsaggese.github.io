#!/bin/bash

# Run Jupyter Notebook in Docker container
# Access at: http://localhost:8888

echo "Starting Jupyter Notebook Server"
echo "=========================================="
echo "URL: http://localhost:8888"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

docker run -it --rm \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
    --name gluonts-jupyter \
    gluonts-covid
