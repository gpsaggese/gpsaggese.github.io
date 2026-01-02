#!/usr/bin/env bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/app \
  -v "$HOME/.kaggle":/root/.kaggle \
  azua-housing:latest \
  python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
