#!/bin/bash
PROJECT_NAME="hmmlearn_anomaly_detection_in_network_traffic"
IMAGE_NAME=$PROJECT_NAME-image
CONTAINER_NAME=$PROJECT_NAME-container

# First perform cleanup to ensure we build the newest image and container
if [ "$(docker images -q $IMAGE_NAME 2> /dev/null)" ]; then
    echo "Image found. Removing: $IMAGE_NAME"
    if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
      echo "Container found. Removing: $CONTAINER_NAME"
      docker container rm $CONTAINER_NAME
    else
       echo "Container '$CONTAINER_NAME' does not exist, skipping removal."
    fi
    docker image rm $IMAGE_NAME
else
    echo "Image '$IMAGE_NAME' does not exist, skipping removal."
fi
docker build -t $IMAGE_NAME .