#!/bin/bash
# """
# Check, for each tutorial under msml610/tutorials/, whether its image has
# been built and whether a container is running, for both the Apple
# `container` tool and Docker.
#
# Usage:
# > msml610/tutorials/check_containers.sh
# """

set -euo pipefail

TUTORIALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf "%-30s %-45s %-12s %-12s %-12s %-12s\n" \
    "TUTORIAL" "IMAGE" "APPLE_BUILT" "APPLE_RUN" "DOCKER_BUILT" "DOCKER_RUN"

for name_file in "$TUTORIALS_DIR"/*/docker_name.sh; do
    tutorial_dir=$(dirname "$name_file")
    tutorial=$(basename "$tutorial_dir")
    # Source in a subshell to avoid var leakage across iterations.
    full_image_name=$(bash -c "source '$name_file' && echo \$FULL_IMAGE_NAME")

    # Apple `container`.
    if container image list --format json 2>/dev/null | \
        grep -q "\"$full_image_name\""; then
        apple_built="yes"
    else
        apple_built="no"
    fi
    if container list --all --format json 2>/dev/null | \
        grep -q "\"$full_image_name\""; then
        apple_running="yes"
    else
        apple_running="no"
    fi

    # Docker.
    if docker images --format '{{.Repository}}' 2>/dev/null | \
        grep -qx "$full_image_name"; then
        docker_built="yes"
    else
        docker_built="no"
    fi
    if docker ps -a --filter "ancestor=$full_image_name" \
        --format '{{.ID}}' 2>/dev/null | grep -q .; then
        docker_running="yes"
    else
        docker_running="no"
    fi

    printf "%-30s %-45s %-12s %-12s %-12s %-12s\n" \
        "$tutorial" "$full_image_name" \
        "$apple_built" "$apple_running" "$docker_built" "$docker_running"
done
