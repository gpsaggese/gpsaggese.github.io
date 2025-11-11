# Start with a basic Ubuntu image.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive

# Update the packages.
RUN apt-get -y update && \
    apt-get -y upgrade

# Install system utilities.
RUN apt install -y --no-install-recommends \
    sudo \
    curl \
    systemctl \
    gnupg \
    git \
    vim

# Install Python.
RUN apt install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev

# Install Python packages.
# There is a bug for tornado==6.2 and jupyter-client==8.0.2. So we pin Tornado and Jupyter.
#   File "/opt/homebrew/lib/python3.10/site-packages/tornado/iostream.py", line 182, in advance
#   assert 0 < size <= self._size
# https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/10
RUN pip3 install \
    dask[complete] \
    fastparquet \
    ipython \
    ipython-sql \
    jupyter \
    jupyter-client==7.3.2 \
    jupyter-contrib-core \
    jupyter-contrib-nbextensions \
    matplotlib \
    numpy \
    pandas \
    psycopg2-binary \
    tornado==6.1 \
    yapf

RUN mkdir /install

# Install Postgres.
RUN apt install -y \
    postgresql \
    postgresql-contrib \
    libpq-dev

# Config.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /var/lib/postgresql/.bashrc

# Set the config for PostgreSQL server.
ADD postgresql.conf pg_hba.conf /install/
RUN cp /install/postgresql.conf /etc/postgresql/14/main
RUN cp /install/pg_hba.conf /etc/postgresql/14/main

# Install graphviz.
RUN apt install -y graphviz
RUN pip3 install graphviz

USER postgres

# Install Jupyter extensions.
ADD install_jupyter_extensions.sh /install/
RUN /install/install_jupyter_extensions.sh

# Report package versions.
#ADD version.sh /install/
#RUN /install/version.sh 2>&1 | tee /data/version.log

# Dask.
EXPOSE 8787
# Postgres.
EXPOSE 5432
# Jupyter.
EXPOSE 8888
