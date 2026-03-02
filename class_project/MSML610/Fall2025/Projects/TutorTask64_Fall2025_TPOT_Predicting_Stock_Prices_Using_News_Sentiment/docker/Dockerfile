# docker/Dockerfile
FROM --platform=$BUILDPLATFORM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl ca-certificates locales \
  && rm -rf /var/lib/apt/lists/*

# Ensure UTF-8 locale (prevents some notebook warnings)
RUN locale-gen en_US.UTF-8 || true
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Copy requirements from repo root (build context is the repo root)
COPY requirements.txt /tmp/requirements.txt

# Install Python deps
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Workdir where your code is mounted at runtime
WORKDIR /workspace

# Jupyter port
EXPOSE 8888

# Default command (usually overridden by docker_run.sh)
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/workspace --ServerApp.default_url=/lab"]
