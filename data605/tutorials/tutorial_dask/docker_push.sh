#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/docker_common/utils.sh

REPO_NAME=gpsaggese
IMAGE_NAME=umd_data605_dask

push_container_image
