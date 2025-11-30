# Use official TFX image as base (has all compatible dependencies pre-installed)
FROM tensorflow/tfx:1.14.0

# Set working directory
WORKDIR /app

# Install additional packages we need
RUN pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipykernel \
    pyyaml \
    tqdm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Override the TFX image's entrypoint to use bash
ENTRYPOINT []
CMD ["/bin/bash"]

# Copy project files
COPY data ./data
COPY utils ./utils
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY pipelines ./pipelines
COPY docs ./docs
COPY CLAUDE.md ./CLAUDE.md
COPY README.md ./README.md

# Create directories for outputs
RUN mkdir -p models pipeline_outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port for Jupyter notebook
EXPOSE 8888

# Expose port for TensorFlow Serving
EXPOSE 8501

# Default command: run bash for interactive use
CMD ["/bin/bash"]
