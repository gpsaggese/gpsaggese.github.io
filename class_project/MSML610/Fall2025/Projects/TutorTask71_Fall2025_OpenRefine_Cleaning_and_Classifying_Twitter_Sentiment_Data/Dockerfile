FROM python:3.11-slim

WORKDIR /app

# Copy your utility modules
COPY utils_data_io.py utils_post_processing.py ./

# Install core deps + BERT stack
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    scikit-learn==1.5.2 \
    jupyter==1.0.0 \
    torch==2.3.0 \
    transformers==4.44.0 \
    datasets==3.0.0 \
    accelerate==0.33.0

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter with no token / password (good for local Docker usage)
CMD ["jupyter", "notebook","--ip=0.0.0.0","--no-browser","--allow-root","--NotebookApp.token=","--NotebookApp.password="]
