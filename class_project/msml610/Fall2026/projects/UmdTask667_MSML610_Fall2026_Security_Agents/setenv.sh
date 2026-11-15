#
# Set up the shell for this repo.
#
# - Activates the project venv
# - Adds the course `helpers` library to `PYTHONPATH`
#
# NOTE: This file needs to be sourced, not executed:
#   > source setenv.sh
#

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
# The course repo is expected next to this repo.
HELPERS_ROOT=${HELPERS_ROOT:-$REPO_DIR/../umd_classes/helpers_root}

if [ ! -d "$HELPERS_ROOT/helpers" ]; then
    echo "ERROR: course helpers not found at '$HELPERS_ROOT'" >&2
    echo "Set HELPERS_ROOT to the 'helpers_root' dir of your umd_classes clone" >&2
    return 1
fi

if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: venv not found; run 'uv venv --python 3.11 .venv' first" >&2
    return 1
fi

source "$REPO_DIR/.venv/bin/activate"
export PYTHONPATH="$REPO_DIR:$HELPERS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "venv=$VIRTUAL_ENV"
echo "PYTHONPATH=$PYTHONPATH"
