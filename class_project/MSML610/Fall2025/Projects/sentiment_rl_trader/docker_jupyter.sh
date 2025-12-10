#!/bin/bash
docker run -it --rm \
    -p 8888:8888 \
    -v $(pwd):/app \
    sentiment_rl_trader \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
