FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

# Install system utilities.
RUN apt install -y --no-install-recommends \
    sudo \
    curl \
    systemctl \
    gnupg \
    git \
    vim \
    build-essential

# Install Python.
RUN apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

RUN python3 -m pip install --upgrade pip
    
# Install Python packages.
# There is a bug:
#   File "/opt/homebrew/lib/python3.10/site-packages/tornado/iostream.py", line 182, in advance
#   assert 0 < size <= self._size
# for tornado==6.2 and jupyter-client==8.0.2.
# https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/10
RUN pip3 install --no-cache-dir \
    --find-links https://storage.googleapis.com/jax-releases/jax_releases.html \
    ipython \
    tornado==6.1 \
    jupyter-client==7.3.2 \
    jupyter-contrib-core \
    jupyter-contrib-nbextensions \
    numpy \
    pandas \
    polars \
    scikit-learn \
    matplotlib \
    seaborn \
    flax \
    optax \
    'jax[cpu]==0.4.13' \
    psycopg2-binary \
    yapf

RUN mkdir /install

# Install Jupyter extensions.
ADD install_jupyter_extensions.sh /install
RUN chmod +x /install/install_jupyter_extensions.sh && /install/install_jupyter_extensions.sh

# Config.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

ENV PYTHONUNBUFFERED=1
WORKDIR /app/project

# Jupyter.
EXPOSE 8888
