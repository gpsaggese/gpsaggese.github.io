#!/bin/bash
# """
# This script launches a Docker container with an interactive bash shell for
# development.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions from the project template.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse default args (-h, -v) and enable set -x if -v is passed.
parse_default_args "$@"

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# List the available Docker images matching the expected image name.
DOCKER_CMD=$(get_docker_cmd)
run "$DOCKER_CMD image ls | grep '$FULL_IMAGE_NAME' || true"

# Configure and run the Docker container with interactive bash shell.
# - Container is removed automatically on exit (--rm)
# - Interactive mode with TTY allocation (-ti)
# - Port forwarding for Jupyter or other services
# - Git root mounted to /git_root inside container
CONTAINER_NAME="${IMAGE_NAME}.bash"
PORT=
# Let the agent runner start each task's container on the host's Docker daemon
# (a sibling of this container), and pass the API key through from the host
# shell without writing it anywhere.
DOCKER_RUN_OPTS="-v /var/run/docker.sock:/var/run/docker.sock -e ANTHROPIC_API_KEY"
DOCKER_CMD=$(get_docker_bash_command)
DOCKER_CMD_OPTS=$(get_docker_bash_options $CONTAINER_NAME "$PORT" "$DOCKER_RUN_OPTS")
# Remove a stale container from an interrupted run and clean up on exit.
kill_existing_container
cleanup_container_on_exit
run "$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME bash"
