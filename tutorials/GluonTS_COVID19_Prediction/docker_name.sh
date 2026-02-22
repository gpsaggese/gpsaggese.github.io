#!/bin/bash
# """
# Docker image naming configuration.
#
# This file defines the repository name, image name, and full image name
# variables used by all docker_*.sh scripts in this project.
# """

REPO_NAME=gpsaggese
# The file should be all lower case.
IMAGE_NAME=umd_gluonts
FULL_IMAGE_NAME=$REPO_NAME/$IMAGE_NAME
