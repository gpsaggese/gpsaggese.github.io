#!/bin/bash
# Run the Docker container on interactive mode with bash access

docker run -it --rm \
    -v $(pwd):/app \
    -p 8888:8888 \
    lime-cnn:latest \
    /bin/bash