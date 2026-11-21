#!/bin/bash
# """
# Execute a command in a Docker container that runs model-generated code.
#
# Same as `docker_cmd.sh`, with these changes:
# - The git root is mounted read-only, except for this project's dir
# - EvalPlus reads its datasets from `data/raw/evalplus_cache`, so scoring
#   works offline
# - The container has no network, unless the first argument is `--ollama`,
#   which lets it reach the Ollama server on the host to generate solutions
# - Each run gets its own container name, so that runs can overlap
#
# Usage:
# > ./docker_cmd_sandbox.sh "python score_coding_runs.py ..."
# > ./docker_cmd_sandbox.sh --ollama "python run_coding_agents.py ..."
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Choose the network: none by default, the host's Ollama server with --ollama.
NETWORK_OPTS="--network none"
if [[ "$1" == "--ollama" ]]; then
    NETWORK_OPTS="-e OLLAMA_URL=http://host.docker.internal:11434"
    shift
fi
CMD="$@"
echo "Executing: '$CMD'"

# Load Docker configuration variables for this script.
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME
print_docker_vars
if [[ "$(get_docker_cmd)" != "docker" ]]; then
    echo "This script needs the docker engine (e.g., CSFY_DOCKER_ENGINE=docker)"
    exit 1
fi

# Find this project's dir inside the container.
PROJECT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PROJECT_IN_CONTAINER=/git_root/${PROJECT_DIR#$GIT_ROOT/}

# Remove the container if this script is interrupted, as `docker_cmd.sh` does.
CONTAINER_NAME=${IMAGE_NAME}.sandbox.$$
trap "docker rm -f $CONTAINER_NAME > /dev/null 2>&1 || true" EXIT INT TERM

# Run the command, with the repo read-only and the project dir writable. Run
# the client in the background and wait for it, so the trap can fire.
docker run --rm -i --name $CONTAINER_NAME \
    $NETWORK_OPTS \
    -v $GIT_ROOT:/git_root:ro \
    -v $PROJECT_DIR:$PROJECT_IN_CONTAINER \
    -e PYTHONPATH=/git_root:/git_root/helpers_root:/git_root/msml610/tutorials \
    -e CSFY_GIT_ROOT_PATH=/git_root \
    -e XDG_CACHE_HOME=$PROJECT_IN_CONTAINER/data/raw/evalplus_cache \
    -w $PROJECT_IN_CONTAINER \
    $FULL_IMAGE_NAME bash -c "$CMD" <&0 &
# Return the exit status of the container command.
wait $!
