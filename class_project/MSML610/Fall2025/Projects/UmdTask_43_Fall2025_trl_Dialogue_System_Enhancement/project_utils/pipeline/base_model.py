"""
load_base_model.py

Utility script to load the original (untrained) base model.
Used for baseline comparisons before RLHF fine-tuning.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_base_model(model_name: str = "microsoft/DialoGPT-small"):
    """
    Load the original base model and tokenizer.

    Args:
        model_name (str): HF model name of the base model.

    Returns:
        tokenizer, model
    """
    print(f"🔹 Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("✅ Base model loaded successfully.\n")
    return tokenizer, model

