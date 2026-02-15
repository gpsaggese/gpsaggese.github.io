#!/bin/bash -xe
# """
# Launch Jupyter Lab server.
#
# This script starts Jupyter Lab on port 8888 with the following configuration:
# - No browser auto-launch (useful for Docker containers)
# - Accessible from any IP address (0.0.0.0)
# - Root user allowed (required for Docker environments)
# - No authentication token or password (for development convenience)
# """

# Start Jupyter Lab with development-friendly settings.
jupyter lab \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password=''

# Alternative: Use classic Jupyter Notebook instead of Jupyter Lab.
#jupyter-notebook \
#    --port=8888 \
#    --no-browser --ip=0.0.0.0 \
#    --allow-root \
#    --NotebookApp.token='' \
#    --NotebookApp.password=''
