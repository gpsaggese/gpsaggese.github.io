#!/bin/bash -xe

REPO_NAME=umd_data605
IMAGE_NAME=umd_data605_template
REPO_NAME=umd_msml610
IMAGE_NAME=umd_msml610_image
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

docker image ls $FULL_IMAGE_NAME