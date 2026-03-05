#!/bin/bash -xe
# """
# Launch Jupyter Lab server inside the Docker container.
#
# Starts Jupyter Lab on port 8888 with no authentication (development mode).
# """

# Start Jupyter Lab with development-friendly settings.
jupyter lab \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password=''
