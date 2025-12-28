"""
Model Loader for DPO-trained T5-large Summarization Model

This module handles loading and initializing the DPO-refined T5-large model
for inference. It provides a singleton pattern to avoid reloading the model
multiple times.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationModel:
    """Singleton class for DPO T5-large model."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model loader (lazy loading)."""
        pass
    
    def load_model(self, model_path=None, device=None):
        """
        Load the DPO-trained T5-large model.
        
        Args:
            model_path: Path to model checkpoint (default: project's DPO model)
            device: Device to load model on (default: auto-detect)
        
        Returns:
            Tuple of (model, tokenizer, device)
        """
        if self._model is not None:
            logger.info("Model already loaded, returning cached instance")
            return self._model, self._tokenizer, self._device
        
        # Set default model path
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "data/models/RLHF-t5-large-merged-dpo"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please ensure the DPO-trained model is available."
            )
        
        # Auto-detect device - Use MPS with CPU fallback for best performance
        # MPS (Apple Silicon GPU) is much faster, with CPU fallback for unsupported ops
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        self._device = device
        
        logger.info(f"Loading DPO T5-large model from {model_path}")
        logger.info(f"Using device: {self._device}")
        
        # Load tokenizer
        self._tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        
        # Load model
        self._model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        self._model.to(self._device)
        self._model.eval()
        
        logger.info("Model loaded successfully")
        
        return self._model, self._tokenizer, self._device
    
    def get_model(self):
        """Get loaded model or load if not already loaded."""
        if self._model is None:
            return self.load_model()
        return self._model, self._tokenizer, self._device
    
    def unload_model(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._device = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Model unloaded from memory")


def load_summarization_model(model_path=None, device=None):
    """
    Convenience function to load the summarization model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    model_loader = SummarizationModel()
    return model_loader.load_model(model_path, device)


def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information
    """
    model_loader = SummarizationModel()
    model, tokenizer, device = model_loader.get_model()
    
    return {
        "model_type": "T5ForConditionalGeneration",
        "model_name": "DPO-trained T5-large",
        "device": str(device),
        "max_length": tokenizer.model_max_length,
        "vocab_size": len(tokenizer),
        "loaded": model is not None
    }


if __name__ == "__main__":
    # Test model loading
    print("Testing model loader...")
    model, tokenizer, device = load_summarization_model()
    print(f"Model loaded on {device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")
    
    # Test inference
    test_text = "summarize: This is a test article about artificial intelligence."
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_length=50)
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test summary: {summary}")
