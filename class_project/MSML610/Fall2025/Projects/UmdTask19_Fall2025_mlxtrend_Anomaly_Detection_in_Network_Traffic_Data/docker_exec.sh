#!/bin/bash
# Open a shell inside the running container
CONTAINER_NAME="anomaly_container"
docker exec -it $CONTAINER_NAME /bin/bash
