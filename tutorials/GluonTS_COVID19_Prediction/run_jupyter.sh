#!/bin/bash -xe

jupyter lab \
    --port=8888 \
    --no-browser --ip=0.0.0.0 \
    --allow-root \
    --ServerApp.token='' --ServerApp.password='' \
    --ServerApp.root_dir='/curr_dir'
