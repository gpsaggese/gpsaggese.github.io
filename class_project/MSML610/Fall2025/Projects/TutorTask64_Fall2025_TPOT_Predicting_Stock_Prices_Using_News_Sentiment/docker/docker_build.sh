#!/usr/bin/env bash
set -euo pipefail

# Absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-umd_msml610_image}"

echo "Building ${IMAGE_NAME}"
echo "  Dockerfile: ${SCRIPT_DIR}/Dockerfile"
echo "  Context:    ${ROOT_DIR}"
docker build --pull --progress=plain \
  -t "${IMAGE_NAME}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${ROOT_DIR}"
