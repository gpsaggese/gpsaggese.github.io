# Use official TFX image as base (has all compatible dependencies pre-installed)
FROM tensorflow/tfx:1.14.0

# Set working directory
WORKDIR /app

# IMPORTANT: TFX is installed in /usr/bin/python (not python3 or conda)
# We must install all packages using /usr/bin/python to keep them together

# Install future (required by tensorflow-transform)
RUN /usr/bin/python -m pip install --no-cache-dir future

# Install our additional packages to the SAME Python as TFX
RUN /usr/bin/python -m pip install --no-cache-dir \
    xgboost==2.0.3 \
    scikit-learn \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm \
    jupyter \
    notebook \
    ipykernel

# Make sure python points to /usr/bin/python (the one with TFX)
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python 1

# Create Jupyter kernel for the correct Python (the one with TFX)
RUN /usr/bin/python -m ipykernel install --name tfx-python --display-name "Python (TFX)" && \
    /usr/bin/python -m ipykernel install --name python3 --display-name "Python 3"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Override the TFX image's entrypoint to use bash
ENTRYPOINT []

# Copy project files
COPY data ./data
COPY utils ./utils
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY pipelines ./pipelines
COPY docs ./docs
COPY CLAUDE.md ./CLAUDE.md
#COPY README.md ./README.md

# Create directories for outputs
RUN mkdir -p models pipeline_outputs

# Expose port for Jupyter notebook
EXPOSE 8888

# Expose port for TensorFlow Serving
EXPOSE 8501

# Default command: run bash for interactive use
CMD ["/bin/bash"]
