#!/bin/bash -xe
# """
# Launch Jupyter Notebook server.
#
# This script starts Jupyter Notebook on port 8888 with the following
# configuration:
# - No browser auto-launch (useful for Docker containers)
# - Accessible from any IP address (0.0.0.0)
# - Root user allowed (required for Docker environments)
# - No authentication token or password (for development convenience)
# """

# Start Jupyter Notebook with development-friendly settings.
jupyter-notebook \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
