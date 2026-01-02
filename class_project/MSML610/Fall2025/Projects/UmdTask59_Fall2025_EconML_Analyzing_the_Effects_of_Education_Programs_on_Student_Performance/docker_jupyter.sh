#!/usr/bin/env bash
set -euo pipefail

IMAGE="umdtask59_econml"
PORT="${PORT:-8888}"

docker run --rm -it \
  -p "${PORT}:8888" \
  -v "$(pwd)":/app \
  "${IMAGE}"
