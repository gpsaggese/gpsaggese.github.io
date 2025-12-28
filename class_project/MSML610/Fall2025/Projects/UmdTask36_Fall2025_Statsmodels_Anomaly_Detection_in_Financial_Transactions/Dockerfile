FROM python:3.10-slim

WORKDIR /workspace

# Install Python dependencies needed for this project
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    imbalanced-learn \
    statsmodels \
    jupyter

EXPOSE 8888

# Start Jupyter Notebook inside the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
