#!/bin/bash -xe

DIR="~/src/venv/client_venv.jupyter_book2"

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

python3 -m venv $DIR
source $DIR/bin/activate
pip install -r $SCRIPT_DIR/requirements.txt
