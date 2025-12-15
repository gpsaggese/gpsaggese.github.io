# Use a standard Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Assuming 'mcp' is the name of the package
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    jupyterlab \
    fastmcp \
    matplotlib \
    seaborn \
    streamlit \
    import-ipynb

# Copy the entire project directory into the container
COPY . .

# Expose the Jupyter port
EXPOSE 8888

# Command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]