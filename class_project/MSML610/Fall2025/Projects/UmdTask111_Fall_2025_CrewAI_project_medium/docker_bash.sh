#!/bin/bash
# Start a bash shell in the Docker container

echo "Starting Docker container with bash shell..."
docker run -it --rm \
    -v "$(pwd):/app" \
    -w /app \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    nba-analysis:latest \
    /bin/bash

