#!/usr/bin/env bash
set -euo pipefail

tag="torchrl_mac:latest"
context_dir="$(dirname "$0")"

echo "Building image ${tag} from ${context_dir}"
docker build -t "${tag}" "${context_dir}"
