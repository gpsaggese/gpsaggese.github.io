# -------------------------------------------------------
# BASE IMAGE
# -------------------------------------------------------
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# -------------------------------------------------------
# SYSTEM DEPENDENCIES
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# WORKSPACE
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# COPY REQUIREMENTS
# -------------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------------
# INSTALL PYTHON DEPENDENCIES
# (Torch already included in base image)
# -------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# CREATE MODEL CACHE DIRECTORY
# -------------------------------------------------------
RUN mkdir -p /app/.cache/huggingface

# -------------------------------------------------------
# PRE-DOWNLOAD MODELS (SAFE)
# Whisper + XLM-R-XNLI + Qwen2.5-1.5B
# -------------------------------------------------------
RUN python - << 'EOF'
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

print("Downloading Whisper-medium...")
whisper.load_model("medium")

print("Downloading XLM-R-XNLI (intent)...")
tok1 = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
mod1 = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

print("Downloading Qwen2.5-1.5B-Instruct (response)...")
tok2 = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True
)
mod2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,
    device_map="cpu"
)

print("All models downloaded successfully!")
EOF

# -------------------------------------------------------
# COPY THE FULL APPLICATION CODE
# -------------------------------------------------------
COPY . .

# -------------------------------------------------------
# RUN THE CHATBOT BY DEFAULT
# -------------------------------------------------------
CMD ["python", "test_chatbot.py"]
