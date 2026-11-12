#!/bin/bash
# """
# Build the Docker image for every tutorial under msml610/tutorials/ by
# calling each tutorial's `docker_build.sh`.
#
# A tutorial that fails to build does not stop the others: every tutorial is
# attempted and a pass/fail summary is printed at the end. Any arguments
# passed to this script (e.g., `-v`, `--no-cache`) are forwarded to each
# `docker_build.sh` call.
#
# Usage:
# > msml610/tutorials/docker_build_all.sh
# > msml610/tutorials/docker_build_all.sh --no-cache
# """

set -euo pipefail

TUTORIALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Track which tutorials succeed / fail to print a summary at the end.
succeeded=()
failed=()

for build_script in "$TUTORIALS_DIR"/*/docker_build.sh; do
    tutorial_dir=$(dirname "$build_script")
    tutorial=$(basename "$tutorial_dir")
    echo "##### Building '$tutorial' #####"
    # `docker_build.sh` builds using the current dir as the build context, so
    # `cd` into the tutorial dir before calling it. Run in a subshell so the
    # `cd` does not leak into the next loop iteration, and so a failure
    # (`set -e` inside `docker_build.sh`) does not abort this loop.
    if ( cd "$tutorial_dir" && ./docker_build.sh "$@" ); then
        succeeded+=("$tutorial")
    else
        echo "ERROR: build failed for '$tutorial'"
        failed+=("$tutorial")
    fi
done

# Report the outcome for every tutorial.
echo
echo "##### Summary #####"
echo "Succeeded (${#succeeded[@]}): ${succeeded[*]:-none}"
echo "Failed (${#failed[@]}): ${failed[*]:-none}"

# Propagate a non-zero exit code if any build failed.
if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
