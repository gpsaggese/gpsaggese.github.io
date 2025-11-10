#!/bin/bash -e

echo "# All images"
docker images

echo "# Images to remove"
docker images

echo "# Removing"
docker image rm -f $(docker images | awk '{ print $3}')

echo "# All images after clean up"
docker images
