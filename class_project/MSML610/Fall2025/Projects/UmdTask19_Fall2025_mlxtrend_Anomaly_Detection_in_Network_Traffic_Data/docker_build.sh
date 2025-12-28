#!/bin/bash
# Build the Docker image
IMAGE_NAME="anomaly-detection"
docker build -t $IMAGE_NAME .
echo "Docker image '$IMAGE_NAME' built successfully!"