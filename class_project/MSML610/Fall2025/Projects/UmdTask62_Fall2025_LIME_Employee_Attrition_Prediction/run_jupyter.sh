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

jupyter-notebook \
    --port=8888 \
    --no-browser --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
