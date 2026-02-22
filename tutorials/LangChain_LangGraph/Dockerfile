FROM ubuntu:24.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

# Install system utilities.
RUN apt install -y --no-install-recommends \
    sudo \
    curl \
    systemctl \
    gnupg \
    git \
    vim

# Install Python and build deps (venv requires python3-venv on Debian/Ubuntu).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment.
RUN python3 -m venv /venv

# Make the venv the default Python.
ENV PATH="/venv/bin:$PATH"

# Upgrade pip inside the venv.
RUN python -m pip install --upgrade pip

# Install Jupyter.
RUN pip3 install jupyterlab

# Install project specific packages.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /install

# Install Jupyter extensions.
#ADD install_jupyter_extensions.sh /install
#RUN /install/install_jupyter_extensions.sh

# Config.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888
