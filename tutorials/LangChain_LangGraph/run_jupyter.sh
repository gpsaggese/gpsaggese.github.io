#!/bin/bash

set -euo pipefail

PORT="${JUPYTER_PORT:-8888}"
VERBOSE=0

while getopts p:v flag
do
    case "${flag}" in
        p) PORT="${OPTARG}";;
        v) VERBOSE=1;;
    esac
done

if [[ "$VERBOSE" == 1 ]]; then
    set -x
fi

jupyter lab \
    --ip=0.0.0.0 \
    --port="$PORT" \
    --no-browser \
    --ServerApp.allow_root=True \
    --ServerApp.token='' \
    --ServerApp.password=''
