#!/bin/bash

PROJECT_NAME="hmmlearn_anomaly_detection_in_network_traffic"
IMAGE_NAME=$PROJECT_NAME-image
CONTAINER_NAME=$PROJECT_NAME-container

if [ "$(docker ps -a -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Container found. Lets ensure it is the newest one."
    docker container rm $CONTAINER_NAME
else
    echo "Container '$CONTAINER_NAME' does not exist, creating and starting it."
fi
docker run -it --name $CONTAINER_NAME -p 8888:8888 $IMAGE_NAME
