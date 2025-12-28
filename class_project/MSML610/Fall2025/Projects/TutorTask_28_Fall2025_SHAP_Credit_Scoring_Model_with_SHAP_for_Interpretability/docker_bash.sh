#!/bin/bash -xe

REPO_NAME=umd_msml610
IMAGE_NAME=umd_msml610_image
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

docker image ls $FULL_IMAGE_NAME

CONTAINER_NAME=${IMAGE_NAME}_bash

docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p 8888:8888 \
    -v $(pwd):/workspace \
    $FULL_IMAGE_NAME \
    bash

# docker run --rm \
# -p 8888:8888 \
# -v $(pwd):/workspace \
# -e PORT=8888 \
# final_project_shap_credit_image \
# bash /workspace/run_jupyter.sh