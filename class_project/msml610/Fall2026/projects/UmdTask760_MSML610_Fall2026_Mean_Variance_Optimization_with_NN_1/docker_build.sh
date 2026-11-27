#!/bin/bash
# Build the unchanged course Dockerfile; keep all build logs in this project.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_DIR"
GIT_ROOT=$(git rev-parse --show-toplevel)
source "$GIT_ROOT/class_project/project_template/utils.sh"
source "$PROJECT_DIR/docker_name.sh"

parse_default_args "$@"
shift $((OPTIND - 1))
DOCKER_CMD=$(get_docker_cmd)
"$DOCKER_CMD" build --progress plain "$@" -t "$FULL_IMAGE_NAME" . 2>&1 | tee docker_build.log
"$DOCKER_CMD" run --rm "$FULL_IMAGE_NAME" /install/version.sh > docker_build.version.log
printf 'SUCCESS: %s built and version check passed.\n' "$FULL_IMAGE_NAME"
