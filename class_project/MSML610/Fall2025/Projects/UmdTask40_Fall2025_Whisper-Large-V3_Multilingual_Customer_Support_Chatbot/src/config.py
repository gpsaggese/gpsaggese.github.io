# src/config.py
import torch
from pathlib import Path

# --- Project Root ---
# This finds the top-level folder of your project
PROJECT_ROOT = Path(__file__).parent.parent 

# --- Device Setup ---
# Checks if you have a GPU (like NVIDIA) or should use the CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Paths ---
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "data" / "sample_audio"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "faqs.json"

# --- Task 2: Transcription ---
WHISPER_MODEL = "openai/whisper-large-v3"

# --- Task 3: Intent Recognition ---
# We'll use a zero-shot model for speed, but you can swap this
# for your fine-tuned BERT model path later.
INTENT_MODEL = "facebook/bart-large-mnli"
INTENT_LABELS = [
    "reset password",
    "check order status",
    "request a refund",
    "billing inquiry",
    "speak to an agent"
]

# --- Task 4: Response Generation ---
# Using a powerful multilingual embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Using a fast, multilingual generative model from Google
LLM_MODEL = "google/flan-t5-large" # (You can swap this for Llama, etc.)

# --- Task 5: Feedback ---
FEEDBACK_LOG_PATH = PROJECT_ROOT / "logs" / "feedback.log"