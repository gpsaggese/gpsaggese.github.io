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

# DAMIAN - Added to fix issue with Scikit-learn and ARM architecture
GOMP_PATH=$(ldconfig -p | grep libgomp)
echo "GOMP_PATH: $GOMP_PATH"
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD


jupyter-notebook \
    --port=8888 \
    --no-browser --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
