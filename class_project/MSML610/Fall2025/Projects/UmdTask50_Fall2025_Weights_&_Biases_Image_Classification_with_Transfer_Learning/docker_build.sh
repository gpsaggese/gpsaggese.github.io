#!/bin/bash -e

# -------------------------------
# Docker build script for local use
# -------------------------------

# Name of your Docker image
IMAGE_NAME=umd_msml610_image

# Get current directory (assumes Dockerfile is here)
BUILD_CONTEXT=$(pwd)

echo "Building Docker image: $IMAGE_NAME"
echo "Using build context: $BUILD_CONTEXT"

# Enable Docker BuildKit (optional but faster)
export DOCKER_BUILDKIT=1

# Build the image
docker build -t $IMAGE_NAME "$BUILD_CONTEXT"

echo "Docker image '$IMAGE_NAME' built successfully!"
