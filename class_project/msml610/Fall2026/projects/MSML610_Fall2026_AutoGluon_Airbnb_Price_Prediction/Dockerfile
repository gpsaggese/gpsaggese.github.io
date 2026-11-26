FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build tools and the OpenMP runtime used by ML libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the project dependencies and notebook tools.
COPY requirements.txt /install/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
    -r /install/requirements.txt \
    jupyterlab "ipykernel<7" jupytext==1.19.3

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
