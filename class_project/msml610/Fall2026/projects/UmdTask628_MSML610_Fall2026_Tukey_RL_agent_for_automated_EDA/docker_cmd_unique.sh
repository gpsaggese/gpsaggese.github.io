#!/bin/bash
# """
# Execute a command in a new Docker container with a unique name.
#
# Same as the template's `docker_cmd.sh`, except the container name: that
# script reuses one name and first kills any running container with it, so
# two concurrent uses (e.g., the tests and an artifact check) kill each other.
# Here every call gets its own name, and only that container is removed on
# exit. The exit status is the command's.
#
# Usage:
# > REL=$(git rev-parse --show-prefix)
# > ./docker_cmd_unique.sh "cd /git_root/$REL && python -m pytest test/test_tukey_env.py"
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

# Configure the container with a name of its own.
CONTAINER_NAME="${IMAGE_NAME}.cmd.$$.${RANDOM}"
DOCKER_CMD=$(get_docker_cmd_command)
DOCKER_CMD_OPTS=$(get_docker_bash_options $CONTAINER_NAME "" "")

# Remove only this container on exit.
cleanup_container_on_exit

# Run the client in the background and wait, as `docker_cmd.sh` does, so
# Ctrl-C reaches the cleanup trap. The command is passed as one argument
# instead of through `run`, which would re-split it on its quotes.
$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME bash -c "$CMD" <&0 &

# Return the exit status of the container command.
wait $!
