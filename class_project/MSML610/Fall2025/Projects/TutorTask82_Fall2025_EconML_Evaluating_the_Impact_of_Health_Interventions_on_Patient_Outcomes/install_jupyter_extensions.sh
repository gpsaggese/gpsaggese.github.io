#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install --no-cache-dir jupyterlab ipywidgets
python3 -m pip check || true
