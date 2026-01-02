#!/bin/bash
# Build the Docker image for NBA Analysis

echo "Building NBA Analysis Docker image..."
docker build -t nba-analysis:latest .
echo "Docker image built successfully!"
echo ""
echo "Next steps:"
echo "  - Run: ./docker_bash.sh (to get a bash shell)"
echo "  - Run: ./docker_jupyter.sh (to start Jupyter)"

