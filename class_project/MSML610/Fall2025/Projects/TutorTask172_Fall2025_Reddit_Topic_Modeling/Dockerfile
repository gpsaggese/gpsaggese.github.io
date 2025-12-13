FROM python:3.11-slim

# System deps required to build fastText + common libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    jupyter \
    fasttext \
    nltk \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm

# Copy project files
COPY . /app

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
