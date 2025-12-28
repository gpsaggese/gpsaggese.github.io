#!/bin/bash

# Build Docker image for GluonTS COVID-19 Project

echo "Building Docker Image: gluonts-covid"
echo "=========================================="

docker build -t gluonts-covid .

if [ $? -eq 0 ]; then
    echo ""
    echo "Docker image built successfully"
    echo ""
    echo "Next steps:"
    echo "  Run Jupyter: ./docker_jupyter.sh"
    echo "  Run bash: ./docker_bash.sh"
else
    echo ""
    echo "Docker build failed"
    exit 1
fi
