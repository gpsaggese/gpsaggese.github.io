#!/bin/bash
# """
# Execute a bash shell in a running Docker container.
#
# This script connects to an already running Docker container and opens an
# interactive bash session for debugging or inspection purposes.
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse default args (-h, -v) and enable set -x if -v is passed.
parse_default_args "$@"

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Execute bash shell in the running container.
exec_container
