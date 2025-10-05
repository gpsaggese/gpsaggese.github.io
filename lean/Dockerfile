# Lean 4 + JupyterLab + sudo + repl
FROM jupyter/minimal-notebook:python-3.11

ENV DEBIAN_FRONTEND=noninteractive

# ---- System deps + sudo (passwordless for jovyan) ----
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo git curl build-essential pkg-config libgmp-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 # passwordless sudo for jovyan
 && echo "jovyan ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/010-jovyan-nopasswd \
 && chmod 0440 /etc/sudoers.d/010-jovyan-nopasswd

# ---- Switch to jovyan for user-space installs ----
USER ${NB_UID}
ENV PATH="/home/jovyan/.elan/bin:${PATH}"

# ---- Install elan (Lean toolchain manager) ----
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y \
 && lean --version

# (Optional) Pin a specific Lean toolchain to avoid surprises, e.g.:
# RUN elan toolchain install v4.11.0 && elan default v4.11.0

# ---- Jupyter kernel: lean4_jupyter ----
RUN pip install --no-cache-dir lean4_jupyter \
 && python -m lean4_jupyter.install \
 && jupyter kernelspec list

# ---- Build the Lean REPL and put it on PATH ----
# Build as jovyan (it uses Lean/Lake toolchain from elan)
RUN git clone https://github.com/leanprover-community/repl.git /home/jovyan/lean-repl \
 && cd /home/jovyan/lean-repl \
 # Ensure toolchain matches the repo's lean-toolchain file
 && elan toolchain install "$(cat lean-toolchain)" \
 && elan default "$(cat lean-toolchain)" \
 && lake build

# Install the compiled repl binary to a root-owned PATH dir
USER root
RUN install -m 0755 /home/jovyan/lean-repl/.lake/build/bin/repl /usr/local/bin/repl

# Verify repl works (the command should print JSON with Lean version)
USER ${NB_UID}
RUN echo '{"cmd":"#eval Lean.versionString"}' | repl

WORKDIR /home/jovyan/work
