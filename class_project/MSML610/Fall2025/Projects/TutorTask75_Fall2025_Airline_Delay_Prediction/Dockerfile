# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8

# Create env from requirements.txt
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Faster, reproducible installs via micromamba
RUN micromamba create -y -n airline-delay-prediction python=3.10 \
 && micromamba run -n airline-delay-prediction pip install --no-cache-dir -r /app/requirements.txt \
 # catboost needs libgomp and openmp runtime; java for LightGBM (feature importances sometimes call out jvm utils)
 && micromamba install -y -n airline-delay-prediction -c conda-forge openjdk=11 libgomp \
 && micromamba clean -a -y

# Add project files
COPY . /app

# Streamlit + Jupyter ports
EXPOSE 8501 8888

# Default shell
SHELL ["/bin/bash", "-lc"]

# Helpful aliases inside the container
RUN echo 'alias act="micromamba run -n airline-delay-prediction"' >> ~/.bashrc

# Default: drop into bash. Use scripts to run Jupyter/Streamlit.
CMD ["bash"]
