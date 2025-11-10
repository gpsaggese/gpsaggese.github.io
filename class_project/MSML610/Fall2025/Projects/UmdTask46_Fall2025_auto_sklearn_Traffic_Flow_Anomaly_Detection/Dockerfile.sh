# Start from the official class image
FROM umd_msml610/umd_msml610_image

# Set the working directory inside the container
WORKDIR /data

# Copy the requirements file into the container first
COPY requirements.txt .

# Install the project-specific libraries from the file
RUN pip install -r requirements.txt

# Expose the Jupyter port
EXPOSE 8888

# Define the command to run when the container starts
# This will start JupyterLab, allowing access to all files
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]