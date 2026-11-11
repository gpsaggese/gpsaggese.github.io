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
# > ./docker_cmd_unique.sh "cd /git_root/$REL && python tukey_evidence.py \
#     check_artifacts --project_dir . --base 10f4d19 --out tmp.checks.json"
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

CMD="$@"
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
CONTAINER_NAME="${IMAGE_NAME}.cmd.$$.${RANDOM}"
DOCKER_CMD=$(get_docker_cmd_command)
DOCKER_CMD_OPTS=$(get_docker_bash_options $CONTAINER_NAME "" "")
cleanup_container_on_exit
# Background the client and wait, as `docker_cmd.sh` does, so Ctrl-C reaches
# the cleanup trap.
$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME bash -c "$CMD" <&0 &
wait $!
