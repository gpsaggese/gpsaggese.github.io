#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker run --rm --init --workdir /workspace --volume "$project_dir:/workspace" \
  --publish 127.0.0.1:8890:8888 quant-ai-course:20261104 \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
