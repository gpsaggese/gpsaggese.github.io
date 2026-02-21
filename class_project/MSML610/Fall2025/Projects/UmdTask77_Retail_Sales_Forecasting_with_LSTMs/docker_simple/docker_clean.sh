#!/bin/bash -e

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/project_template/utils.sh

REPO_NAME=msml610
IMAGE_NAME=retail_sales_forecasting_lstms

remove_container_image
