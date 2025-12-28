#!/usr/bin/env bash
set -e

echo "============================================"
echo "Building Docker image: causal_success_analysis"
echo "============================================"

docker build -t causal_success_analysis .

echo "============================================"
echo "Build complete!"
echo "Image name: causal_success_analysis"
echo "============================================"
