#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

REPO_NAME=umd_msml610
IMAGE_NAME=onnx_timeseries_forecasting

push_container_image
