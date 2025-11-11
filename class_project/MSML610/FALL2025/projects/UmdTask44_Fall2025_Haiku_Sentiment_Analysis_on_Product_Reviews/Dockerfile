# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn nltk jupyter

# Expose Jupyter port
EXPOSE 8888

# Launch Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
