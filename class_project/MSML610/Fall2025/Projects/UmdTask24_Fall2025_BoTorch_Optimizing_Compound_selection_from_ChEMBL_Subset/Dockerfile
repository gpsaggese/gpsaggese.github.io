# BoTorch Multi-Objective Drug Discovery Docker Image
# ====================================================
#
# This Dockerfile is configured to work with the following files:
#   - BoTorch.API.md
#   - BoTorch.API.ipynb
#   - Botorch.example.md
#   - Botorch.example.ipynb
#   - botorch_utils.py
#   - Dockerfile
#
# Usage:
#   docker build -t botorch-drug-discovery .
#   docker run -p 8888:8888 -v $(pwd):/workspace botorch-drug-discovery
#
# Then open http://localhost:8888 in your browser

FROM python:3.10-slim

LABEL description="BoTorch Multi-Objective Optimization for Drug Discovery"
LABEL version="1.0.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
# Core ML/optimization stack
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    botorch==0.9.2 \
    gpytorch==1.11 \
    pyro-ppl==1.8.6

# Scientific computing
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.1 \
    scikit-learn==1.3.0

# Cheminformatics
RUN pip install --no-cache-dir \
    rdkit==2023.3.2

# Visualization
RUN pip install --no-cache-dir \
    matplotlib==3.7.2 \
    seaborn==0.12.2

# Jupyter environment
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    jupyterlab==4.0.3 \
    ipywidgets==8.1.0

# Additional utilities
RUN pip install --no-cache-dir \
    tqdm==4.65.0 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0

# Copy project files (using exact names from screenshot)
COPY Botorch_utils.py /workspace/
COPY BoTorch.API.md /workspace/
COPY BoTorch.API.ipynb /workspace/
COPY Botorch.example.md /workspace/
COPY Botorch.example.ipynb /workspace/
COPY Dockerfile /workspace/

# Create directories for data and results
RUN mkdir -p /workspace/data \
    /workspace/results \
    /workspace/notebooks

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
ENV PYTHONPATH=/workspace

# Expose Jupyter port
EXPOSE 8888

# Create startup script
RUN echo '#!/bin/bash\n\
echo "========================================"\n\
echo "BoTorch Drug Discovery Environment"\n\
echo "========================================"\n\
echo ""\n\
echo "Available files:"\n\
echo "  • botorch_utils.py        - Wrapper utilities"\n\
echo "  • BoTorch.API.md          - API documentation"\n\
echo "  • BoTorch.API.ipynb       - API examples"\n\
echo "  • Botorch.example.md      - Usage examples"\n\
echo "  • Botorch.example.ipynb   - Interactive examples"\n\
echo ""\n\
echo "Starting Jupyter Lab..."\n\
echo "Open http://localhost:8888 in your browser"\n\
echo ""\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --NotebookApp.token="" --NotebookApp.password=""\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.allow_root = True\n\
c.NotebookApp.ip = '0.0.0.0'\n\
c.NotebookApp.open_browser = False\n\
c.NotebookApp.port = 8888\n\
c.NotebookApp.notebook_dir = '/workspace'\n\
" > /root/.jupyter/jupyter_notebook_config.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8888 || exit 1

# Default command
CMD ["/workspace/start.sh"]

# ====================================================
# Build Instructions:
# ====================================================
#
# Prerequisites:
#   Ensure you have these files in the same directory:
#     ✓ botorch_utils.py
#     ✓ BoTorch.API.md
#     ✓ BoTorch.API.ipynb
#     ✓ Botorch.example.md
#     ✓ Botorch.example.ipynb
#     ✓ Dockerfile (this file)
#
# 1. Build the image:
#    docker build -t botorch-drug-discovery .
#
# 2. Run with volume mount (to persist data):
#    docker run -p 8888:8888 \
#      -v $(pwd)/data:/workspace/data \
#      -v $(pwd)/results:/workspace/results \
#      botorch-drug-discovery
#
# 3. Run interactively (with shell access):
#    docker run -it --rm -p 8888:8888 \
#      -v $(pwd):/workspace \
#      botorch-drug-discovery /bin/bash
#
# 4. Run with GPU support (if available):
#    docker run --gpus all -p 8888:8888 \
#      -v $(pwd):/workspace \
#      botorch-drug-discovery
#
# 5. Run with custom Jupyter password:
#    docker run -p 8888:8888 \
#      -e JUPYTER_TOKEN="your-secret-token" \
#      botorch-drug-discovery
#
# ====================================================
# File Structure in Container:
# ====================================================
#
# /workspace/
# ├── botorch_utils.py          # Wrapper utilities (32 KB)
# ├── BoTorch.API.md            # API documentation (24 KB)
# ├── BoTorch.API.ipynb         # API examples notebook (17 KB)
# ├── Botorch.example.md        # Usage examples (24 KB)
# ├── Botorch.example.ipynb     # Interactive examples (1.5 MB)
# ├── Dockerfile                # This file (5 KB)
# ├── data/                     # Data directory (mount your data here)
# ├── results/                  # Results directory
# └── notebooks/                # Additional notebooks
#
# ====================================================
# Quick Start After Container Runs:
# ====================================================
#
# 1. Open browser to http://localhost:8888
#
# 2. Start with one of these notebooks:
#    • BoTorch.API.ipynb       - Learn the API
#    • Botorch.example.ipynb   - See complete examples
#
# 3. Import utilities in any notebook:
#    from botorch_utils import ChemDataProcessor, MultiObjectiveOptimizer
#
# 4. Read documentation:
#    • BoTorch.API.md          - Full API reference
#    • Botorch.example.md      - Usage patterns
#
# ====================================================
# Environment Variables:
# ====================================================
#
# Customize behavior with environment variables:
#
#   docker run -e BOTORCH_DEVICE=cuda \
#              -e BOTORCH_SEED=42 \
#              -e JUPYTER_TOKEN=mysecret \
#              -p 8888:8888 \
#              botorch-drug-discovery
#
# Available variables:
#   - BOTORCH_DEVICE: 'cpu' or 'cuda' (default: cpu)
#   - BOTORCH_SEED: Random seed (default: 42)
#   - JUPYTER_PORT: Port for Jupyter (default: 8888)
#   - JUPYTER_TOKEN: Security token (default: empty)
#   - PYTHONPATH: Python import path (default: /workspace)
#
# ====================================================
# Common Docker Commands:
# ====================================================
#
# • List running containers:
#   docker ps
#
# • Stop container:
#   docker stop <container_id>
#
# • View logs:
#   docker logs <container_id>
#
# • Execute command in running container:
#   docker exec -it <container_id> bash
#
# • Copy files from container:
#   docker cp <container_id>:/workspace/results ./local_results
#
# • Remove container:
#   docker rm <container_id>
#
# • Remove image:
#   docker rmi botorch-drug-discovery
#
# ====================================================
# Troubleshooting:
# ====================================================
#
# 1. Port already in use:
#    Solution: Change port
#    docker run -p 8889:8888 botorch-drug-discovery
#
# 2. Permission issues with mounted volumes:
#    Solution: Run with user permissions
#    docker run --user $(id -u):$(id -g) -p 8888:8888 botorch-drug-discovery
#
# 3. Out of memory:
#    Solution: Increase Docker memory limit
#    docker run --memory=8g --memory-swap=8g -p 8888:8888 botorch-drug-discovery
#
# 4. Missing files error:
#    Solution: Ensure all 6 files are in build directory
#    ls -la | grep -E "(botorch_utils|BoTorch|Botorch|Dockerfile)"
#
# 5. Jupyter not accessible:
#    Solution: Check container is running
#    docker ps
#    docker logs <container_id>
#
# 6. Python import errors:
#    Solution: Verify PYTHONPATH
#    docker exec -it <container_id> python -c "import sys; print(sys.path)"
#
# ====================================================
# Advanced Usage:
# ====================================================
#
# • Run with custom Python version:
#   Change FROM line to: FROM python:3.11-slim
#
# • Add more dependencies:
#   Add RUN pip install <package> before COPY commands
#
# • Use different base image (with CUDA):
#   FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
#
# • Multi-stage build (smaller image):
#   Use separate builder and runtime stages
#
# • Persist Jupyter config:
#   -v $(pwd)/jupyter_config:/root/.jupyter
#
# ====================================================
# Docker Compose Alternative:
# ====================================================
#
# Create docker-compose.yml:
#
# version: '3.8'
# services:
#   botorch:
#     build: .
#     ports:
#       - "8888:8888"
#     volumes:
#       - ./data:/workspace/data
#       - ./results:/workspace/results
#     environment:
#       - BOTORCH_DEVICE=cpu
#       - BOTORCH_SEED=42
#
# Then run:
#   docker-compose up
#
# ====================================================
