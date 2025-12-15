#!/bin/bash

set -e

echo "Building Docker image..."
docker build -f Dockerfile.mcp -t bert-fake-news-mcp:latest .
echo "Build complete. Run ./docker_run.sh to start the server."
