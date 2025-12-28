# Use a standard, public Python image (slim version is smaller and faster)
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /data

# Install system dependencies required for auto-sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]