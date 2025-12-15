#!/bin/bash
echo "Building Docker container..."
cd "$(dirname "$0")"
docker build -t electricity-forecasting:latest .
if [ $? -eq 0 ]; then
    echo "✅ Build successful! Run: ./docker/run_jupyter.sh"
else
    echo "❌ Build failed!"
    exit 1
fi
