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

# Import tmux utils and rename window.
source $GIT_ROOT/helpers_root/dev_scripts_helpers/thin_client/thin_client_utils.sh
OLD_TMUX_TITLE=$(tmux_rename_on_entry "jupy" || true)

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
CONTAINER_NAME="${IMAGE_NAME}.jupyter"
# Kill existing container if -f flag is set.
kill_existing_container_if_forced

DOCKER_ENGINE_CURRENT=$(get_docker_engine)
DOCKER_CMD=$(get_docker_cmd)

if [[ "$DOCKER_ENGINE_CURRENT" == "apple" ]]; then
    # Apple container's -p port forwarding is broken (v1.0.0), so we skip it
    # and print instructions for a manual port forward script.
    echo "Apple container engine detected."
    echo "NOTE: Apple's container tool has a bug where -p port forwarding does"
    echo "not work. To access Jupyter from your browser, run this in another"
    echo "terminal after the container starts:"
    echo ""
    echo "  docker_jupyter_port_forward.sh $CONTAINER_NAME $JUPYTER_HOST_PORT"
    echo ""
    DOCKER_CMD_OPTS=$(get_docker_jupyter_options $CONTAINER_NAME $JUPYTER_HOST_PORT $JUPYTER_USE_VIM)
    # Run container in detached mode (we follow logs instead of -ti).
    run "$DOCKER_CMD run --rm -d $DOCKER_CMD_OPTS $FULL_IMAGE_NAME $CMD"
    sleep 3
    CONTAINER_IP=$(get_container_ip $CONTAINER_NAME)
    if [[ -n "$CONTAINER_IP" ]]; then
        echo "Container IP: $CONTAINER_IP"
        echo "Direct URL: http://$CONTAINER_IP:$JUPYTER_HOST_PORT"
        echo ""
    fi
    # Follow logs so the user sees Jupyter output.
    $DOCKER_CMD logs -f $CONTAINER_NAME
else
    DOCKER_CMD_OPTS=$(get_docker_jupyter_options $CONTAINER_NAME $JUPYTER_HOST_PORT $JUPYTER_USE_VIM)
    DOCKER_CMD=$(get_docker_jupyter_command)
    run "$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME $CMD"
fi

# Restore the TMUX.
tmux_restore_on_exit "$OLD_TMUX_TITLE"
