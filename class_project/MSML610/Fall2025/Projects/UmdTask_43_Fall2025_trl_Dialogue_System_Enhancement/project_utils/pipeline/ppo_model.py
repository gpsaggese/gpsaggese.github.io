"""
load_finetuned_model.py

Utility script to load the RLHF-enhanced fine-tuned model from HuggingFace Hub.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_finetuned_model(model_repo: str = "VenkataSivaRajesh/Rlhf_Enhanced_DialoGpt"):
    """
    Load the fine-tuned model and tokenizer from HuggingFace Hub.

    Args:
        model_repo (str): HuggingFace repository name of the fine-tuned model.

    Returns:
        tokenizer, model
    """
    print(f"🔹 Loading fine-tuned model from repo: VenkataSivaRajesh/Rlhf_Enhanced_DialoGpt")

    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForCausalLM.from_pretrained(model_repo)

    print("✅ Fine-tuned model loaded successfully.\n")
    return tokenizer, model

