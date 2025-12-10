#!/usr/bin/env bash

# Resolve project root (one level up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -it \
  -p 8888:8888 \
  -v "$PROJECT_ROOT":/workspace \
  msml610_fashion \
  bash
