#!/usr/bin/env bash
docker run --rm -it \
  -p 8888:8888 -p 8000:8000 \
  -v "$(pwd)":/app \
  -v "$HOME/.kaggle":/root/.kaggle \
  azua-housing:latest bash
