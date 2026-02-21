#!/bin/bash -e
# """
# Launch Jupyter Lab server.
#
# This script starts Jupyter Lab on port 8888 with the following configuration:
# - No browser auto-launch (useful for Docker containers)
# - Accessible from any IP address (0.0.0.0)
# - Root user allowed (required for Docker environments)
# - No authentication token or password (for development convenience)
# """

# Parse params.
export JUPYTER_HOST_PORT=8888
export JUPYTER_USE_VIM=0
export TARGET_DIR=""
export VERBOSE=0

OLD_CMD_OPTS=$@
while getopts p:d:uv flag
do
    case "${flag}" in
        p) JUPYTER_HOST_PORT=${OPTARG};;
        u) JUPYTER_USE_VIM=1;;
        d) TARGET_DIR=${OPTARG};;
        # /Users/saggese/src/git_gp1/code/
        v) VERBOSE=1;;
    esac
done

if [[ $VERBOSE == 1 ]]; then
    set -x
fi;

# Disable announcements extension (JupyterLab 3.2+).
jupyter labextension disable @jupyterlab/apputils-extension:announcements 2>/dev/null || true

# Install vim extension for JupyterLab if requested.
if [[ $JUPYTER_USE_VIM != 0 ]]; then
    # Try to install jupyterlab-vim extension
    pip install jupyterlab-vim 2>/dev/null || true
    jupyter labextension install @axlair/jupyterlab_vim 2>/dev/null || true
fi;

# Create Jupyter Server configuration (used by JupyterLab).
mkdir -p ~/.jupyter
cat << EOT >> ~/.jupyter/jupyter_server_config.py
#------------------------------------------------------------------------------
# Jupytext Configuration for JupyterLab
#------------------------------------------------------------------------------
# Always pair ipynb notebooks to py files
c.ContentsManager.default_jupytext_formats = "ipynb,py"
# Use the percent format when saving as py
c.ContentsManager.preferred_jupytext_formats_save = "py:percent"
c.ContentsManager.outdated_text_notebook_margin = float("inf")

#------------------------------------------------------------------------------
# Autosave Configuration
#------------------------------------------------------------------------------
# Autosave interval in seconds (60 seconds = 1 minute)
c.ServerApp.autosave_interval = 60
EOT

# Start Jupyter Lab with development-friendly settings.
jupyter lab \
    --port=$JUPYTER_HOST_PORT \
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
