#!/bin/bash -e

echo "# All images"
docker images

echo "# Images to remove"
docker images | egrep 'umd_data605|ubuntu'

echo "# Removing"
docker image rm $(docker images | egrep 'umd_data605|ubuntu' | awk '{ print $3}')

echo "# All images after clean up"
docker images
