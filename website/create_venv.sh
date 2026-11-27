#!/usr/bin/env bash
# """
# Create the virtual environment with all the Python packages needed to run the
# scripts under `website/`, using `website/requirements.txt` and the
# `jupyter_book/requirements.txt` of each book. Re-running the script rebuilds
# the venv from scratch, e.g., after a Homebrew upgrade of Python leaves the
# venv pointing to a missing interpreter.
#
# Usage:
# > ./website/create_venv.sh
# > source ~/src/venv/client_venv.website/bin/activate
#
# - Use a specific Python interpreter:
#   > PYTHON=python3.11 ./website/create_venv.sh
# """
set -euo pipefail

GIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$HOME/src/venv/client_venv.website"
PYTHON="${PYTHON:-python3}"

echo "==> Creating venv $VENV_DIR"
"$PYTHON" -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing Python packages"
pip install --upgrade pip
# Install everything with one command so that pip resolves all the
# constraints together.
pip install \
  -r "$GIT_ROOT/website/requirements.txt" \
  -r "$GIT_ROOT/data605/jupyter_book/requirements.txt" \
  -r "$GIT_ROOT/msml610/jupyter_book/requirements.txt" \
  -r "$GIT_ROOT/tutorials/jupyter_book/requirements.txt"

# Browser used by `blog_to_substack.py --tables png`.
echo "==> Installing Playwright browser"
playwright install chromium

# Report the tools that pip can't install.
for tool in git node pandoc prettier; do
  if ! command -v "$tool" >/dev/null; then
    echo "WARNING: '$tool' not found: install it with 'brew install $tool'" >&2
  fi
done

cat <<EOF
==> Done. Activate the venv with:
  > source $VENV_DIR/bin/activate
EOF
