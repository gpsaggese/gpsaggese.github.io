# Use a slim Python image for efficiency and stability
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python dependencies file
COPY requirements.txt .

# Install Python packages defined in requirements.txt (Fairlearn, pandas, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Jupyter Notebooks
EXPOSE 8888

# Copy all project files into the container (including your .py and .ipynb files)
COPY . /app

# Set default command to bash for easy execution and debugging
CMD ["bash"]