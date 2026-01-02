FROM --platform=$BUILDPLATFORM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

# Install system utilities.
RUN apt install -y --no-install-recommends \
    sudo \
    curl \
    gnupg \
    git \
    vim \
    software-properties-common

# Install Python 3.10+
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip
    
# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /install

# Install common packages
ADD install_common_packages.sh /install/
RUN chmod +x /install/install_common_packages.sh && /install/install_common_packages.sh

# Install Jupyter
RUN pip3 install --no-cache-dir jupyter jupyterlab notebook

# Set working directory
WORKDIR /data

# Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
