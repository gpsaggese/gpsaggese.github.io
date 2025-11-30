#!/bin/bash
# Script to build and run the TFX pipeline in Docker

set -e  # Exit on error

echo "========================================================================"
echo "  House Price TFX Pipeline - Docker Runner"
echo "========================================================================"

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Function to build Docker image
build_image() {
    echo "Building Docker image..."
    docker-compose -f docker/docker-compose.yml build
    echo "[OK] Docker image built successfully!"
}

# Function to start container
start_container() {
    echo "Starting Docker container..."
    docker-compose -f docker/docker-compose.yml up -d
    echo "[OK] Container started!"
}

# Function to run pipeline
run_pipeline() {
    echo "Running TFX pipeline inside container..."
    docker exec house-price-tfx python scripts/api.py
}

# Function to stop container
stop_container() {
    echo "Stopping Docker container..."
    docker-compose -f docker/docker-compose.yml down
    echo "[OK] Container stopped!"
}

# Function to show logs
show_logs() {
    docker-compose -f docker/docker-compose.yml logs -f
}

# Function to enter container
enter_container() {
    echo "Entering container (type 'exit' to leave)..."
    docker exec -it house-price-tfx bash
}

# Main menu
case "${1:-help}" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    run)
        run_pipeline
        ;;
    stop)
        stop_container
        ;;
    logs)
        show_logs
        ;;
    bash|shell)
        enter_container
        ;;
    full)
        echo "Running full workflow: build -> start -> run pipeline"
        build_image
        start_container
        sleep 3
        run_pipeline
        ;;
    help|*)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  start     - Start the Docker container"
        echo "  run       - Run the TFX pipeline"
        echo "  stop      - Stop the Docker container"
        echo "  logs      - Show container logs"
        echo "  bash      - Enter container shell"
        echo "  full      - Build, start, and run pipeline (all-in-one)"
        echo "  help      - Show this help message"
        echo ""
        echo "Example workflow:"
        echo "  $0 build      # Build the image"
        echo "  $0 start      # Start container"
        echo "  $0 run        # Run pipeline"
        echo "  $0 bash       # Enter container to explore"
        echo "  $0 stop       # Stop when done"
        echo ""
        echo "Or simply:"
        echo "  $0 full       # Do everything at once"
        ;;
esac

echo ""
echo "========================================================================"
