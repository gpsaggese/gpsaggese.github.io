FROM python:3.11-slim

WORKDIR /work

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt

# Copy project
COPY . /work

# Jupyter Lab on container start
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password=", "--allow-root"]
