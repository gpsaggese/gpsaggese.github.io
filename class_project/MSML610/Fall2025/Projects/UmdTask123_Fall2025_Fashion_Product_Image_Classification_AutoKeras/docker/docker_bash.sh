#!/usr/bin/env bash

HOST_DIR="C:/Users/konda/src/umd_classes/class_project/MSML610/Fall2025/Projects/UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras"

docker run --rm -it \
  -p 8888:8888 \
  -v "$HOST_DIR":/workspace \
  msml610_fashion \
  bash
