#!/usr/bin/env bash
set -euo pipefail

tag="torchrl_mac:latest"
context_dir="$(dirname "$0")"

# Build if missing
if ! docker image inspect "${tag}" >/dev/null 2>&1; then
  "${context_dir}/docker_build.sh"
fi

echo "Starting Jupyter Lab container (image: ${tag})"
echo "Access at: http://localhost:8888"
echo "Press Ctrl+C to stop"
docker run -it --rm \
  -p 8888:8888 \
  -v "${context_dir}:/app" \
  "${tag}"
