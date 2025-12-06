#!/usr/bin/env bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  msml610_autokeras \
  bash
