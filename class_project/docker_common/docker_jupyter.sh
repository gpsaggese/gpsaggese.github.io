#!/bin/bash
#
# Execute run_jupyter.sh in the container.
# 
# Usage:
# > docker_jupyter.sh -d /Users/saggese/src/git_gp1/code/book.2018.Martin.Bayesian_Analysis_with_Python.2e -v -u -p 8889
#
set -e
#set -x
# Parse params.
export JUPYTER_HOST_PORT=8888
export JUPYTER_USE_VIM=0
export TARGET_DIR=""
export VERBOSE=0
OLD_CMD_OPTS=$@
while getopts p:d:uv flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;
        u) JUPYTER_USE_VIM=1;;
        d) TARGET_DIR=${OPTARG};;
        # /Users/saggese/src/git_gp1/code/
        v) VERBOSE=1;;
    esac
done
if [[ $VERBOSE == 1 ]]; then
    set -x
fi;

# Import the utility functions.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh
source $GIT_ROOT/class_project/docker_common/utils.sh

# Execute the script setting the vars for this tutorial.
get_docker_vars_script ${BASH_SOURCE[0]}
@@ -58,4 +58,4 @@ run "docker run \
    $DOCKER_RUN_OPTS \
    -v $(pwd):/curr_dir \
    $FULL_IMAGE_NAME \
    $CMD"
    $CMD"