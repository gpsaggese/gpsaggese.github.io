#!/bin/bash
# Run the Docker container with Jupyter notebook

docker run -it --rm \
    -v $(pwd):/app \
    -p 8888:8888 \
    lime-cnn:latest