FROM --platform=$BUILDPLATFORM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# Basic system update
RUN apt-get -y update && \
    apt-get -y upgrade

# Install system utilities
RUN apt-get install -y --no-install-recommends \
    sudo \
    curl \
    systemctl \
    gnupg \
    git \
    vim

# Install Python
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev

RUN python3 -m pip install --upgrade pip

# Extra build deps for scientific Python stacks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /install

# Install common packages (course script)
ADD install_common_packages.sh /install/
RUN chmod +x /install/install_common_packages.sh && /install/install_common_packages.sh

# Install Jupyter extensions (course script)
ADD install_jupyter_extensions.sh /install/
RUN chmod +x /install/install_jupyter_extensions.sh
RUN /install/install_jupyter_extensions.sh

# Install project-specific Python requirements
COPY requirements.txt /install/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /install/requirements.txt

# Shell / sudo config
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions (course script)
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Expose Jupyter port
EXPOSE 8888
