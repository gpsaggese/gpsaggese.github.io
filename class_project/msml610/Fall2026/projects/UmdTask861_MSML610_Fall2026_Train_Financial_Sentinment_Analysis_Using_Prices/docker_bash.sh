#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
opts=(--rm --init --workdir /workspace --volume "$project_dir:/workspace")
if [[ -t 0 && -t 1 ]]; then opts+=(-it); fi
docker run "${opts[@]}" quant-ai-course:20261104 bash "$@"
