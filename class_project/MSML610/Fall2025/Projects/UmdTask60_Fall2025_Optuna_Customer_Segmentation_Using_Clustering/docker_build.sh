#!/bin/bash -e

# Docker Build Script for MSML610 Optuna Customer Segmentation
# Builds the Docker image for the project

GIT_ROOT=$(git rev-parse --show-toplevel)

source $GIT_ROOT/class_project/docker_common/utils.sh

REPO_NAME=umd_msml610

IMAGE_NAME=umd_msml610_image

# Build container with Docker BuildKit
export DOCKER_BUILDKIT=1

#export DOCKER_BUILDKIT=0


echo "Building Docker image: ${REPO_NAME}/${IMAGE_NAME}"


build_container_image

echo ""
echo "Docker image built successfully!"
echo "Image: ${REPO_NAME}/${IMAGE_NAME}"
echo ""
echo "Next: Run with: bash docker_jupyter.sh"
echo ""
