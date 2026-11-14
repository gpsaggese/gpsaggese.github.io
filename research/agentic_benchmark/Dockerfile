# Use Python 3.12 slim (already has Python and pip).
FROM python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates (needed for HTTPS).
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install project specific packages.
RUN mkdir -p /install
COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyterlab jupyterlab_vim jupytext -r /install/requirements.txt

# Config.
COPY etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
COPY version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888

CMD ["/bin/bash"]
