#!/usr/bin/env bash
set -euo pipefail

IMAGE="umdtask59_econml"

docker run --rm -it \
  -v "$(pwd)":/app \
  "${IMAGE}" bash
