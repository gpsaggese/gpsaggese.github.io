# Copy one of the Dockerfile style and customize it.

# #############################################################################
# Python 3.12 slim (already has Python and pip).
# #############################################################################

FROM python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates (needed for HTTPS) and git.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install project specific packages.
RUN mkdir -p /install
COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyterlab jupyterlab_vim jupytext -r /install/requirements.txt

# Config.
COPY etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
COPY version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888

# #############################################################################
# Ubuntu 24.04 + pip to install.
# #############################################################################

# FROM ubuntu:24.04
# ENV DEBIAN_FRONTEND noninteractive

# # Install system utilities and Python in a single layer.
# RUN apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y --no-install-recommends \
#       sudo \
#       curl \
#       git \
#       build-essential \
#       python3 \
#       python3-pip \
#       python3-dev \
#       python3-venv \
#     && rm -rf /var/lib/apt/lists/*

# # Create virtual environment.
# RUN python3 -m venv /opt/venv

# # Make the venv the default Python.
# ENV PATH="/opt/venv/bin:$PATH"

# # Install project specific packages.
# RUN mkdir /install
# COPY requirements.txt /install/requirements.txt
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir jupyterlab jupyterlab_vim jupytext -r /install/requirements.txt

# # Config.
# COPY etc_sudoers /install/
# COPY etc_sudoers /etc/sudoers
# COPY bashrc /root/.bashrc

# # Report package versions.
# COPY version.sh /install/
# RUN /install/version.sh 2>&1 | tee version.log

# # Jupyter.
# EXPOSE 8888

# #############################################################################
# Ubuntu 24.04 + uv.
# #############################################################################

# FROM ubuntu:24.04
# ENV DEBIAN_FRONTEND noninteractive

# # Install system utilities and Python in a single layer.
# RUN apt-get update && \
#   apt-get upgrade -y && \
#   apt-get install -y --no-install-recommends \
#     sudo \
#     curl \
#     git \
#     build-essential \
#     python3 \
#     python3-pip \
#     python3-dev \
#     python3-venv \
#     libgomp1 \
#     g++ \
#   && rm -rf /var/lib/apt/lists/*

# # Install uv for package management.
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ENV PATH="/root/.local/bin:$PATH"

# # Install project specific packages using uv.
# COPY pyproject.toml uv.lock /app/
# WORKDIR /app
# RUN uv sync
# ENV PATH="/app/.venv/bin:$PATH"

# # Install Jupyter.
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir jupyterlab jupyterlab_vim jupytext

# # Copy project files.
# COPY . /app

# RUN mkdir /install

# # Config.
# COPY etc_sudoers /install/
# COPY etc_sudoers /etc/sudoers
# COPY bashrc /root/.bashrc

# # Report package versions.
# COPY version.sh /install/
# RUN /install/version.sh 2>&1 | tee version.log

# # Jupyter.
# EXPOSE 8888
