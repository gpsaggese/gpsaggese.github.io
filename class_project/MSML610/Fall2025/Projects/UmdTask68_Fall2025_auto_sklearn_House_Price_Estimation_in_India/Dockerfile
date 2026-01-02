FROM --platform=linux/amd64 python:3.9-slim

# Install system dependencies required for auto-sklearn and other packages
# swig is often needed for auto-sklearn dependencies; build-essential for compiling C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies one by one to handle complex dependencies
# Install numpy and cython first as they are build deps for many packages
RUN pip install --no-cache-dir "numpy<2.0.0" cython

# Install scikit-learn compatible with auto-sklearn 0.15.0
# auto-sklearn 0.15.0 requires scikit-learn<0.25.0,>=0.24.0
RUN pip install --no-cache-dir "scikit-learn>=0.24.0,<0.25.0"

# Install other dependencies
RUN pip install --no-cache-dir pandas matplotlib seaborn jupyterlab>=4.1 xgboost joblib

# Finally install auto-sklearn
RUN pip install --no-cache-dir "auto-sklearn==0.15.0"

# Copy the rest of the project
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Default command to run jupyter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
