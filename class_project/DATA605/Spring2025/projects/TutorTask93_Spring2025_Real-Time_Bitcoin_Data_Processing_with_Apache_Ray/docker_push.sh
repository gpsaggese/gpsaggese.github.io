#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/tutorial_github_simple/project_template/utils.sh

REPO_NAME=umd_data605
IMAGE_NAME=bitcoin_cli_project

push_container_image
