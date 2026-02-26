FROM ubuntu:24.04
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

# Install uv, if needed.
#RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#ENV PATH="/root/.local/bin:$PATH"
#COPY pyproject.toml uv.lock /app/
#RUN uv sync
    
# Install Jupyter.
RUN pip3 install jupyterlab

# Install project specific packages.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /install

# Install Postgres.
RUN apt-get update && apt-get install -y \
    postgresql \
    postgresql-contrib \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Config Postgres.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /var/lib/postgresql/.bashrc
ADD postgresql.conf pg_hba.conf /install/
RUN cp /install/postgresql.conf /etc/postgresql/16/main
RUN cp /install/pg_hba.conf /etc/postgresql/16/main

# Install graphviz.
RUN apt-get update && apt-get install -y graphviz && rm -rf /var/lib/apt/lists/*
RUN pip3 install graphviz

# Report package versions.
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee /install/version.log

USER postgres

# Dask.
EXPOSE 8787
# Postgres.
EXPOSE 5432
# Jupyter.
EXPOSE 8888
