# =========================
# Stage 1: Builder
# =========================
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Install build tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
RUN /opt/venv/bin/python -m pip install --upgrade pip

# Install uv package manager
RUN pip install uv

# Copy frozen requirements
COPY requirements.txt /install/requirements.txt

# Install all dependencies using uv
RUN uv pip install --no-cache-dir -r /install/requirements.txt


# =========================
# Stage 2: Final image
# =========================
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Runtime deps only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv and workspace from builder
COPY . /workspace/
COPY --from=builder /opt/venv /opt/venv

# Copy configs
RUN mkdir -p /install
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc
COPY version.sh /install/version.sh

# Run version report
RUN /install/version.sh 2>&1 | tee version.log

# Workdir and expose Jupyter
WORKDIR /workspace
EXPOSE 8888
