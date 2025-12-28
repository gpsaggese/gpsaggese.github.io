#!/bin/bash
# Start Jupyter notebook in the Docker container

echo "Starting Jupyter notebook..."
echo "Jupyter will be available at: http://localhost:8888"
echo "Check the token in the output below"
echo ""

docker run -it --rm \
    -p 8888:8888 \
    -v "$(pwd):/app" \
    -w /app \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    nba-analysis:latest \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

