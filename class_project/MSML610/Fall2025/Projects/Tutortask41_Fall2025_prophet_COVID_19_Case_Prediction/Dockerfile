FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip && \
    pip install \
        pandas \
        numpy \
        matplotlib \
        prophet \
        pmdarima \
        scikit-learn \
        tensorflow \
        keras \
        jupyter

# Copy project files
COPY . /app

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["sh","-lc","jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"]
