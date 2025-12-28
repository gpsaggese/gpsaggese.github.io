#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh

REPO_NAME=msml610
IMAGE_NAME=retail_sales_forecasting_lstms

exec_container
