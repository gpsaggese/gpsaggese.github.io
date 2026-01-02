#!/usr/bin/env bash
set -e

IMAGE_NAME="xgboost-docker-image"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .
