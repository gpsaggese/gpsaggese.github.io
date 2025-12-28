#!/bin/bash

if [ "$1" = "build" ]; then
    echo "Building Docker image..."
    docker build -t covid-causal-analysis .
    
elif [ "$1" = "run" ]; then
    echo "Running analysis in Docker..."
    docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" covid-causal-analysis
    
elif [ "$1" = "shell" ]; then
    echo "Opening interactive shell in Docker..."
    docker run -it --rm -v "$(pwd):/app" -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" covid-causal-analysis /bin/bash
    
elif [ "$1" = "jupyter" ]; then
    echo "Starting Jupyter notebook..."
    docker run -d --rm -p 8888:8888 -v "$(pwd):/app" -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" --name covid-jupyter covid-causal-analysis jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
    echo "Jupyter running at: http://localhost:8888"
    
elif [ "$1" = "stop" ]; then
    echo "Stopping containers..."
    docker stop $(docker ps -q --filter "name=covid") 2>/dev/null || true
    
elif [ "$1" = "clean" ]; then
    echo "Cleaning up..."
    docker system prune -f
    docker rmi covid-causal-analysis 2>/dev/null || true
    
else
    echo "Usage: $0 {build|run|shell|jupyter|stop|clean}"
    echo ""
    echo "Commands:"
    echo "  build   - Build Docker image"
    echo "  run     - Run analysis once"
    echo "  shell   - Open interactive shell"
    echo "  jupyter - Start Jupyter notebook"
    echo "  stop    - Stop all containers"
    echo "  clean   - Clean Docker system"
    exit 1
fi
