#!/bin/bash
# """
# Execute Jupyter Lab in a Docker container.
#
# This script launches a Docker container running Jupyter Lab with
# configurable port, directory mounting, and vim bindings. It passes
# command-line options to the run_jupyter.sh script inside the container.
#
# Usage:
# > docker_jupyter.sh [options]
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse command-line options and set Jupyter configuration variables.
parse_docker_jupyter_args "$@"

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# List available Docker images and inspect architecture.
list_and_inspect_docker_image

# Run the Docker container with Jupyter Lab.
CMD=$(get_run_jupyter_cmd "${BASH_SOURCE[0]}" "$OLD_CMD_OPTS")
CONTAINER_NAME=$IMAGE_NAME
# Kill existing container if -f flag is set.
kill_existing_container_if_forced

DOCKER_CMD=$(get_docker_jupyter_command)
DOCKER_CMD_OPTS=$(get_docker_jupyter_options $CONTAINER_NAME $JUPYTER_HOST_PORT $JUPYTER_USE_VIM)
run "$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME $CMD"
