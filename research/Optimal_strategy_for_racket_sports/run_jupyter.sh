#!/bin/bash
# """
# Launch Jupyter Lab server.
#
# This script starts Jupyter Lab on port 8888 with the following configuration:
# - No browser auto-launch (useful for Docker containers)
# - Accessible from any IP address (0.0.0.0)
# - Root user allowed (required for Docker environments)
# - No authentication token or password (for development convenience)
# - Vim keybindings can be enabled via JUPYTER_USE_VIM environment variable
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Print each command to stdout before executing it.
#set -x

# Import the utility functions from /git_root.
GIT_ROOT=/git_root
source $GIT_ROOT/class_project/project_template/utils.sh

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars

# Setup Jupyter Lab environment.
setup_jupyter_environment

# Initialize Jupyter Lab command with base configuration.
JUPYTER_ARGS=$(get_jupyter_args)

# Start Jupyter Lab with development-friendly settings.
run "jupyter lab $JUPYTER_ARGS"
