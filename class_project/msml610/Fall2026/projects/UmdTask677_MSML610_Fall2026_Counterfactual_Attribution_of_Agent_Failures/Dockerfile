# Copy one of the Dockerfile style and customize it.

# #############################################################################
# Python 3.12 slim (already has Python and pip).
# #############################################################################

# Build for x86-64 on every host. The official evaluator depends on
# `ytsaurus-yson`, which ships Linux wheels only for x86-64, and the SWE-rebench
# task images are x86-64 too. On an Apple Silicon Mac this image runs under
# emulation.
FROM --platform=linux/amd64 python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates (needed for HTTPS), git, curl, and basic build tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install the Docker CLI only, without a daemon. The agent runner starts each
# task's container through the host's Docker daemon, whose socket the
# `docker_*.sh` scripts mount into this container.
ARG DOCKER_CLI_VERSION=27.5.1
RUN curl -fsSL "https://download.docker.com/linux/static/stable/$(uname -m)/docker-${DOCKER_CLI_VERSION}.tgz" \
    | tar -xz -C /usr/local/bin --strip-components=1 docker/docker

# Install project specific packages. Pin jupytext to avoid jitter in the
# notebook files due to changing version. Pin ipykernel below 7 since 7.3.0
# randomly stops answering execute requests, which hangs `jupyter nbconvert
# --execute` forever.
RUN mkdir -p /install
COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyterlab jupyterlab_vim "ipykernel<7" jupytext==1.19.3 -r /install/requirements.txt

# Install the official SWE-rebench evaluator (the SWE-bench fork used by
# SWE-rebench) in its own virtual environment, so its dependencies cannot clash
# with the runner's. Pin the commit and the dependency versions that scored all
# the results so far.
ARG SWEBENCH_FORK_COMMIT=e4907b7a90eafaa1f0a6428fd04fe31cdd8b4284
COPY requirements.eval.txt /install/requirements.eval.txt
RUN python -m venv /opt/swebench && \
    /opt/swebench/bin/pip install --no-cache-dir \
      -c /install/requirements.eval.txt \
      "swebench @ git+https://github.com/SWE-rebench/SWE-bench-fork.git@${SWEBENCH_FORK_COMMIT}"
ENV CAAF_EVAL_PYTHON=/opt/swebench/bin/python

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
