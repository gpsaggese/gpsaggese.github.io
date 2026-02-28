FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Jupyter Lab.
RUN pip install --upgrade pip && \
    pip install jupyterlab

# Install project-specific requirements.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy configuration files.
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report installed package versions for debugging.
COPY version.sh /version.sh
RUN bash /version.sh

EXPOSE 8888
