#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8888}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-/curr_dir}"

cd "$NOTEBOOK_DIR"

exec jupyter lab   --ip=0.0.0.0   --port="$PORT"   --no-browser   --allow-root   --ServerApp.token=''   --ServerApp.password=''   --ServerApp.disable_check_xsrf=True   --ServerApp.allow_remote_access=True
