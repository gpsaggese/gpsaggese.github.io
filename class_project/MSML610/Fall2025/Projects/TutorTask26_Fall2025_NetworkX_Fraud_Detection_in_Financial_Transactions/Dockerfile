# Use a lightweight Python base
FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Copy everything into the container
COPY . /workspace

# Install Python dependencies if requirements.txt exists
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Install Jupyter
RUN pip install notebook

# Expose the Jupyter default port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
