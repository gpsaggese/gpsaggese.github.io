#!/bin/bash -xe

REPO_NAME=gpsaggese
IMAGE_NAME=umd_data605_packages
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME

docker image ls $FULL_IMAGE_NAME

HOST_PORT=8889

CONTAINER_NAME=$IMAGE_NAME
docker run --rm -ti \
    --name $CONTAINER_NAME \
    -p $HOST_PORT:8888 -p 5432:5432 \
    -v $(pwd):/data \
    $FULL_IMAGE_NAME
