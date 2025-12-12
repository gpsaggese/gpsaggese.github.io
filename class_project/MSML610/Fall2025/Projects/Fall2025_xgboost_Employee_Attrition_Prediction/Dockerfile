FROM python:3.10-slim

# Install basic system deps (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /workspace

# Copy requirements and install them + Jupyter
COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
    jupyter

# Expose Jupyter port
EXPOSE 8888

# Default command: open a bash shell (scripts will override this if needed)
CMD ["/bin/bash"]
