#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-umd_msml610_image}"
CONTAINER_NAME="${CONTAINER_NAME:-umd_msml610_dev}"
PORT="${PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"  # leave empty while debugging

# Clean up any old container with the same name
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8888" \
  -v "${ROOT_DIR}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc "jupyter lab \
              --ip=0.0.0.0 \
              --port=8888 \
              --no-browser \
              --allow-root \
              --ServerApp.root_dir=/workspace \
              --ServerApp.default_url=/lab \
              --ServerApp.allow_remote_access=True \
              ${JUPYTER_TOKEN:+--ServerApp.token=${JUPYTER_TOKEN}}"
