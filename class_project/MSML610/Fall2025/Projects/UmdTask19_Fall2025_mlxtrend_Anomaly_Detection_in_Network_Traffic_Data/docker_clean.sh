#!/bin/bash
# Stop and remove all containers, and delete dangling images
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker system prune -f
echo "Cleaned up Docker environment."
