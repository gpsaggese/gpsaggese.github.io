#!/bin/bash -xe
if [ -d "/data" ]; then
    cd /data
    echo "Starting Jupyter in /data (mounted host folder)"
elif [ -d "/curr_dir" ]; then
    cd /curr_dir
    echo "Starting Jupyter in /curr_dir (fallback)"
else
    cd /
    echo "Starting Jupyter in / (fallback)"
fi

if ! command -v jupyter-notebook &> /dev/null; then
    echo "Error: jupyter-notebook command not found."
    echo "Are you running this script inside the docker container?"
    echo "You should probably run ./docker_scripts/docker_jupyter.sh instead."
    exit 1
fi

jupyter-notebook \
    --port=8888 \
    --no-browser --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
