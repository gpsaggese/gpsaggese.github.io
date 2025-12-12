# ===========================
# Credit Card Fraud Detection - Docker Setup
# ===========================
FROM python:3.11-slim

WORKDIR /work

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .

# Install core requirements (jupyterlab==3.*, ipywidgets==7.* already included).
RUN pip install --no-cache-dir -r requirements.txt

# --- What-If Tool setup (ensure WIT + TensorFlow CPU stack) ---
RUN pip install --no-cache-dir tensorflow-cpu==2.20.0 witwidget

# ---- Jupyter setup ----
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
