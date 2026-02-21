#!/bin/bash
# """
# Execute Jupyter Lab in a Docker container.
#
# This script launches a Docker container running Jupyter Lab with configurable
# port, directory mounting, and vim bindings. It passes command-line options to
# the run_jupyter.sh script inside the container.
#
# Usage:
# > docker_jupyter.sh -d /path/to/notebooks -v -u -p 8889
# """

# Exit immediately if any command exits with a non-zero status.
set -e
#set -x

# Initialize default parameter values for Jupyter configuration.
export JUPYTER_HOST_PORT=8888
export JUPYTER_USE_VIM=0
export TARGET_DIR=""
TARGET_DIR=.
export VERBOSE=0

# Save original command-line options to pass through to run_jupyter.sh.
OLD_CMD_OPTS=$@

# Parse command-line options.
while getopts p:d:uv flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;  # Port for Jupyter Lab
        u) JUPYTER_USE_VIM=1;;            # Enable vim bindings
        d) TARGET_DIR=${OPTARG};;         # Directory to mount as /data
        v) VERBOSE=1;;                    # Enable verbose output
    esac
done

# Enable command tracing if verbose mode is requested.
if [[ $VERBOSE == 1 ]]; then
    set -x
fi;

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Configure Docker run options with port forwarding and optional volume mount.
# We use the same port inside and outside the container, so that the localhost
# printed inside the container is the correct one.
DOCKER_RUN_OPTS="-p $JUPYTER_HOST_PORT:8888"
if [[ $TARGET_DIR != "" ]]; then
    DOCKER_RUN_OPTS="$DOCKER_RUN_OPTS -v $TARGET_DIR:/data"
fi;
CMD="/curr_dir/run_jupyter.sh $OLD_CMD_OPTS"

# List available Docker images and inspect architecture.
run "docker image ls $FULL_IMAGE_NAME"
(docker manifest inspect $FULL_IMAGE_NAME | grep arch) || true

# Run the Docker container with Jupyter Lab.
CONTAINER_NAME=$IMAGE_NAME
run "docker run \
    --rm -ti \
    --name $CONTAINER_NAME \
    $DOCKER_RUN_OPTS \
    -v $(pwd):/curr_dir \
    -v $GIT_ROOT:/git_root \
    -e PYTHONPATH=/git_root:/git_root/helpers_root \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    $FULL_IMAGE_NAME \
    $CMD"
