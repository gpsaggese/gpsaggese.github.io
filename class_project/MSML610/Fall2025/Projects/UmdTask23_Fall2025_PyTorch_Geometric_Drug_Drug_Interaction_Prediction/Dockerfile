FROM mambaorg/micromamba:latest

WORKDIR /workspace

# Copy project into container (micromamba image runs as non-root user by default)
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

# Install everything in the base env (CPU versions)
RUN micromamba install -y -n base -c conda-forge -c pytorch -c pyg \
    python=3.11 \
    numpy pandas scikit-learn matplotlib \
    jupyter notebook \
    rdkit \
    pytorch cpuonly \
    pyg \
    && micromamba clean -a -y

CMD ["python", "-c", "print('Container ready. Use docker run ... jupyter notebook to launch.')"]
