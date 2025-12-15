#!/bin/bash

# Build Docker image for Sentiment Analysis project
echo "Building Docker image..."
docker build -t umd_msml610_sentiment:latest .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    echo "Image name: umd_msml610_sentiment:latest"
else
    echo "✗ Docker build failed"
    exit 1
fi