# Python 3.12 slim image from the class project template (Python and pip are
# already installed). All Python packages come as wheels, so no compiler is
# needed.
FROM python:3.12-slim

# Avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install CA certificates (needed for HTTPS when refreshing the data) and git.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the project packages. jupytext and ipykernel are pinned as in the
# template: jupytext to avoid jitter in the paired notebook files, and
# ipykernel below 7 because 7.3.0 can hang `jupyter nbconvert --execute`.
RUN mkdir -p /install
COPY requirements.txt /install/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir jupyterlab jupyterlab_vim "ipykernel<7" jupytext==1.19.3 -r /install/requirements.txt

# Config.
COPY etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
COPY version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888
