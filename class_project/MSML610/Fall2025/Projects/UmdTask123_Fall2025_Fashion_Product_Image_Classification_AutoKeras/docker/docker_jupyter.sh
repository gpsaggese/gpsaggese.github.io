#!/usr/bin/env bash

# NOTE: This path is specific to *your* machine.
# For the TA, they can edit HOST_DIR to their own path or just run docker manually.

HOST_DIR="C:/Users/konda/src/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras"

docker run --rm \
  -p 8888:8888 \
  -v "$HOST_DIR":/workspace \
  msml610_fashion \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
