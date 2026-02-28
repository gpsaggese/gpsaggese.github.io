# Use Python 3.12 slim (already has Python and pip).
FROM python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    curl \
    gnupg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Neo4j.
RUN curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/neo4j.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" \
    | tee /etc/apt/sources.list.d/neo4j.list \
    && apt-get update && apt-get install -y neo4j \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment.
RUN python3 -m venv /opt/venv

# Make the venv the default Python.
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside the venv.
RUN python -m pip install --upgrade pip

# Install Jupyter.
RUN pip3 install jupyterlab

# Install project specific packages.
COPY requirements.txt /install/requirements.txt
RUN pip3 install --no-cache-dir -r /install/requirements.txt

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
