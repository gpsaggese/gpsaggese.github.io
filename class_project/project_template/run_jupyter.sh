#!/bin/bash -xe

jupyter lab \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password=''

#jupyter-notebook \
#    --port=8888 \
#    --no-browser --ip=0.0.0.0 \
#    --allow-root \
#    --NotebookApp.token='' \
#    --NotebookApp.password=''
