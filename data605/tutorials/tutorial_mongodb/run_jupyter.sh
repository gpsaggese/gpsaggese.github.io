#!/bin/bash -xe

/data/run_mongo.sh || true
# Enable vim bindings.
#jupyter nbextension enable vim_binding/vim_binding
jupyter-notebook --port=8888 --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --allow-root
