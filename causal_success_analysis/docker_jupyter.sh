#!/bin/bash

echo "Starting Jupyter Lab server..."
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/app \
    causal_success_analysis

echo "Jupyter Lab is running at http://localhost:8888"
