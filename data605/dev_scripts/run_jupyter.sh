#!/bin/bash -xe

# Enable vim bindings.
jupyter nbextension enable vim_binding/vim_binding
jupyter-notebook \
    --port=8888 \
    --no-browser --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
