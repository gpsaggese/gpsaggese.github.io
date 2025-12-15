#!/bin/bash

# Run Docker container for Sentiment Analysis project
echo "Starting Jupyter notebook server..."
echo ""

docker run -it --rm \
    -p 8889:8888 \
    -v $(pwd):/data \
    --name sentiment_analysis \
    umd_msml610_sentiment:latest

echo ""
echo "Container stopped."