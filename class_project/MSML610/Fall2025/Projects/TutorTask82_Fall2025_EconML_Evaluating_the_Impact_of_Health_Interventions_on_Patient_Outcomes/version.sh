#!/usr/bin/env bash
set -euo pipefail
echo "=== Python ==="
python3 --version
echo "=== Pip ==="
python3 -m pip --version
echo "=== Installed packages (top 200) ==="
python3 -m pip list --format=columns | head -n 200
