#!/bin/bash
# Run container interactively
IMAGE_NAME="anomaly-detection"
CONTAINER_NAME="anomaly_container"
docker run -it --name $CONTAINER_NAME -p 8888:8888 $IMAGE_NAME