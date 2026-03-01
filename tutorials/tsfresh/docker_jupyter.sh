#!/bin/bash
# """
# Launch Jupyter Lab inside the tsfresh Docker container.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Default options.
PORT=8888
TARGET_DIR="."
VIM_BINDINGS=0
VERBOSE=0

# Parse command-line options.
while getopts "p:d:uv" opt; do
    case $opt in
        p) PORT=$OPTARG ;;
        d) TARGET_DIR=$OPTARG ;;
        u) VIM_BINDINGS=1 ;;
        v) VERBOSE=1 ;;
        *) echo "Usage: $0 [-p port] [-d dir] [-u] [-v]" && exit 1 ;;
    esac
done

# Import the utility functions from the project template.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Configure volume mount if a target directory is specified.
VOLUME_OPTS=""
if [[ "$TARGET_DIR" != "." ]]; then
    VOLUME_OPTS="-v $TARGET_DIR:/data"
fi

CONTAINER_NAME=${IMAGE_NAME}_jupyter

cmd="docker run \
    --rm \
    --name $CONTAINER_NAME \
    -p $PORT:8888 \
    -v $(pwd):/curr_dir \
    -v $GIT_ROOT:/git_root \
    $VOLUME_OPTS \
    -e PYTHONPATH=/git_root:/git_root/helpers_root \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    $FULL_IMAGE_NAME \
    bash /run_jupyter.sh"
run $cmd
