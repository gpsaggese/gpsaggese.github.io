# Set project root
GIT_ROOT=$(git rev-parse --show-toplevel)

# Optional: Load shared utils if you have them
if [ -f "$GIT_ROOT/project_template/utils.sh" ]; then
    source "$GIT_ROOT/project_template/utils.sh"
fi

# Define Docker image and container names for reuse
export IMAGE_NAME="pyarrow-btc-project"
export CONTAINER_NAME="pyarrow-btc-container"
