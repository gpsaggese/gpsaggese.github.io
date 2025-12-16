#!/usr/bin/env bash
set -euo pipefail
IMAGE="task22_clip:latest"

docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/work \
  -w /work \
  -e PYTHONPATH=/work \
  "$IMAGE" \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
