#!/bin/bash

# Run interactive bash shell in Docker container

echo "Starting Interactive Bash Shell"
echo "=========================================="
echo "Working directory: /workspace"
echo "Type 'exit' to leave"
echo "=========================================="
echo ""

docker run -it --rm \
    -v "$(pwd)":/workspace \
    -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
    --name gluonts-bash \
    gluonts-covid \
    /bin/bash
