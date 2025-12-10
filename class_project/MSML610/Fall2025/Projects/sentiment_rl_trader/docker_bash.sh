#!/bin/bash
docker run -it --rm \
    -v $(pwd):/app \
    sentiment_rl_trader /bin/bash
