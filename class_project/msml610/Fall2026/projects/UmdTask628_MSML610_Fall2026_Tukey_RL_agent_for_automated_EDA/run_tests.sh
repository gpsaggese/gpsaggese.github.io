#!/bin/bash
# """
# Run the Tukey unit tests inside the project container.
#
# The tests are passed as explicit file paths: the repo-level
# `class_project/msml610/conftest.py` ignores test collection under that tree
# when pytest is given a directory. `TUKEY_RUN_TESTS=1` marks an intentional
# run. pytest exits with code 5 when it collects nothing, which fails this
# script. Extra arguments are passed to pytest, e.g. `./run_tests.sh -k scoring`.
#
# Usage:
# > ./run_tests.sh
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Locate the project dir inside the container (the git root is mounted at
# /git_root).
GIT_ROOT=$(git rev-parse --show-toplevel)
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL_DIR=${PROJECT_DIR#"$GIT_ROOT"/}

# Run the unit tests in a container with a unique name (`docker_cmd.sh` reuses
# one name and kills any running container with it), re-quoting the extra
# arguments so that e.g. `-k "env or sets"` reaches pytest as two arguments.
cd "$PROJECT_DIR"
PYTEST_ARGS=$(printf '%q ' "$@")
./docker_cmd_unique.sh "cd /git_root/$REL_DIR && TUKEY_RUN_TESTS=1 python -m pytest test/test_tukey_*.py $PYTEST_ARGS"
