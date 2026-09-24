#!/bin/bash
# """
# Execute Jupyter Lab in a Docker container.
#
# This script launches a Docker container running Jupyter Lab with
# configurable port, directory mounting, and vim bindings. It passes
# command-line options to the run_jupyter.sh script inside the container.
#
# By default, Jupyter Lab opens directly to this script's own directory
# (e.g., http://localhost:8888/lab/tree/git_root/<path/to/this/dir>). Pass
# -r to open at Jupyter Lab's root instead.
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

# Directory Jupyter Lab should open to by default (repo root if -r was passed).
JUPYTER_TARGET_DIR=$(get_jupyter_target_dir "${BASH_SOURCE[0]}" $JUPYTER_OPEN_ROOT)
if [[ -n "$JUPYTER_TARGET_DIR" ]]; then
    JUPYTER_URL_PATH="/lab/tree${JUPYTER_TARGET_DIR}"
else
    JUPYTER_URL_PATH="/lab"
fi

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
    # Apple container's -p port forwarding is broken (v1.0.0), so we start
    # a local port-forward tunnel ourselves instead of asking the user to
    # run it by hand in another terminal.
    echo "Apple container engine detected."
    DOCKER_CMD_OPTS=$(get_docker_jupyter_options $CONTAINER_NAME $JUPYTER_HOST_PORT $JUPYTER_USE_VIM "$JUPYTER_TARGET_DIR")
    # Run container in detached mode (we follow logs instead of -ti).
    run "$DOCKER_CMD run --rm -d $DOCKER_CMD_OPTS $FULL_IMAGE_NAME $CMD"
    sleep 3
    CONTAINER_IP=$(get_container_ip $CONTAINER_NAME)
    if [[ -n "$CONTAINER_IP" ]]; then
        echo "Container IP: $CONTAINER_IP"
        echo "Direct URL: http://$CONTAINER_IP:$JUPYTER_HOST_PORT${JUPYTER_URL_PATH}"
        echo ""
    fi
    # echo "To access Jupyter from your browser, run this in another terminal "
    # echo "after the container starts:"
    # echo "> docker_jupyter_port_forward.py $CONTAINER_NAME $JUPYTER_HOST_PORT"
    # Start the tunnel in the background and make sure it is killed whenever
    # this script exits, for any reason (normal exit, error, Ctrl+C).
    docker_jupyter_port_forward.py "$CONTAINER_NAME" "$JUPYTER_HOST_PORT" &
    TUNNEL_PID=$!
    trap 'kill $TUNNEL_PID 2>/dev/null' EXIT INT TERM
    echo "Tunnel started (PID $TUNNEL_PID): http://localhost:$JUPYTER_HOST_PORT${JUPYTER_URL_PATH}"
    echo ""
    # Follow logs so the user sees Jupyter output.
    $DOCKER_CMD logs -f $CONTAINER_NAME
else
    DOCKER_CMD_OPTS=$(get_docker_jupyter_options $CONTAINER_NAME $JUPYTER_HOST_PORT $JUPYTER_USE_VIM "$JUPYTER_TARGET_DIR")
    DOCKER_CMD=$(get_docker_jupyter_command)
    run "$DOCKER_CMD $DOCKER_CMD_OPTS $FULL_IMAGE_NAME $CMD"
fi

# Restore the TMUX.
tmux_restore_on_exit "$OLD_TMUX_TITLE"
