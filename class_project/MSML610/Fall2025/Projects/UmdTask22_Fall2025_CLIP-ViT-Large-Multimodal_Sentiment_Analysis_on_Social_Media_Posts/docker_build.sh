#!/usr/bin/env bash
set -euo pipefail
IMAGE="task22_clip:latest"
docker build -t "$IMAGE" .
echo "Built $IMAGE"
