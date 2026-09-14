# Use Python 3.12 slim (already has Python and pip).
FROM python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates, git, build tools, and graphviz for DAG visualization.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment.
RUN python3 -m venv /opt/venv

# Make the venv the default Python.
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside the venv.
RUN python -m pip install --upgrade pip

# Install project specific packages.
COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade pip 
# Pin down jupytext to avoid jitter in the notebook files due to changing
# version.
RUN pip install --no-cache-dir jupyterlab jupyterlab_vim jupytext==1.19.3 -r /install/requirements.txt

RUN mkdir -p /install

# Config.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888
