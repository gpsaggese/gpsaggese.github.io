FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# xformers binary wheels are often available for specific torch/cuda combos. 
# We try installing via pip which should fetch the wheel if available or build from source (which is slow).
# Using --extra-index-url for pytorch if needed, but we are in a pytorch image.
RUN pip install --no-cache-dir \
    xformers \
    yfinance \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyter \
    ipykernel

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Expose port for Jupyter
EXPOSE 8888

# Command to run Jupyter notebook by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
