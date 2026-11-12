#!/bin/bash
# """
# Execute a command in a Docker container.
#
# This script runs a specified command inside a new Docker container instance.
# The container is removed automatically after the command completes. The
# git root is mounted to /git_root inside the container.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse default args (-h, -v) and enable set -x if -v is passed.
# Shift processed option flags so remaining args form the command.
parse_default_args "$@"
shift $((OPTIND-1))

# Capture the command to execute from remaining arguments.
CMD="$@"
echo "Executing: '$CMD'"

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# List available Docker images matching the expected image name.
DOCKER_CMD=$(get_docker_cmd)
run "$DOCKER_CMD image ls | grep '$FULL_IMAGE_NAME' || true"
#(docker manifest inspect $FULL_IMAGE_NAME | grep arch) || true

# Configure and run the Docker container with the specified command.
CONTAINER_NAME="${IMAGE_NAME}.cmd"
DOCKER_CMD=$(get_docker_cmd_command)
PORT=""
DOCKER_RUN_OPTS=""
DOCKER_CMD_OPTS=$(get_docker_bash_options $CONTAINER_NAME $PORT $DOCKER_RUN_OPTS)
# Remove a stale container from an interrupted run and clean up on exit.
kill_existing_container
cleanup_container_on_exit
# Run the client in the background and wait for it. The Apple engine client
# fails to forward Ctrl-C to the container and keeps running, so a foreground
# run would never reach the cleanup trap. `wait` returns as soon as the trap
# fires, and the trap removes the container, which also ends the client. Keep
# stdin attached (`<&0`) since a background job otherwise reads /dev/null.
run "$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME bash -c '$CMD' <&0 &"
# Return the exit status of the container command.
wait $!
