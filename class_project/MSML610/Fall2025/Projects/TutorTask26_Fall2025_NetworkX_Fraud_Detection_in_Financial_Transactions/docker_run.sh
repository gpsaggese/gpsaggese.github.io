#!/bin/bash

IMAGE_NAME="networkx-image"
CONTAINER_NAME="networkx-container"

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Stopping any existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Running Jupyter Notebook container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  $IMAGE_NAME

echo "Container started."
echo "Access it at: http://localhost:8888"
echo "Check container logs for token:"
echo "    docker logs $CONTAINER_NAME"
