FROM python:3.11-slim

# -----------------------------
# 1. System dependencies
# -----------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 2. Environment settings
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# -----------------------------
# 3. Install Python libraries
# -----------------------------
RUN pip install --upgrade pip

# Core data libs
RUN pip install \
    pandas \
    numpy \
    scikit-learn \
    seaborn \
    matplotlib \
    imbalanced-learn \
    nltk

# Install PyTorch CPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# HuggingFace ecosystem
RUN pip install \
    transformers \
    datasets \
    accelerate \
    tokenizers \
    peft

# Explainability
RUN pip install shap lime

# Notebook environment
RUN pip install jupyter

# -----------------------------
# 4. Download nltk data inside the container
# -----------------------------
RUN python - <<EOF
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
EOF

# -----------------------------
# 5. Copy project files
# -----------------------------
COPY . /app

EXPOSE 8888

# -----------------------------
# 6. Start Jupyter
# -----------------------------
CMD ["sh","-lc","jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'"]
