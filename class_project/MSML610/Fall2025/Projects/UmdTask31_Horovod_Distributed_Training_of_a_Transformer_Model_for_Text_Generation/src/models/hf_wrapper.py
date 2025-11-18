"""
Hugging Face model wrapper for pretrained models.

Supports loading and fine-tuning DistilGPT-2 and GPT-2.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config
)


class HFModelWrapper(nn.Module):
    """
    Wrapper for Hugging Face causal language models.
    
    Provides a unified interface compatible with our custom TransformerLM.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        pad_token_id: int = 50256
    ):
        """
        Initialize HF model wrapper.
        
        Args:
            model_name: Pretrained model name (e.g., 'gpt2', 'distilgpt2').
            cache_dir: Cache directory for models.
            pad_token_id: Padding token ID.
        """
        super().__init__()
        
        self.model_name = model_name
        self.pad_token_id = pad_token_id
        
        # Load pretrained model
        print(f"[INFO] Loading pretrained model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Store config
        self.config = self.model.config
        self.vocab_size = self.config.vocab_size
        
        print(f"[INFO] Model loaded successfully:")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Vocab size: {self.vocab_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with our training loop.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
            labels: Target labels for loss computation (batch, seq_len).
            
        Returns:
            Tuple of (logits, loss).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using HF's generate method.
        
        Args:
            input_ids: Starting token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            top_p: Nucleus sampling.
            eos_token_id: End-of-sequence token ID.
            **kwargs: Additional arguments for HF generate.
            
        Returns:
            Generated token IDs.
        """
        self.eval()
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                do_sample=True,
                pad_token_id=self.pad_token_id,
                **kwargs
            )
        
        return output

    # Gradient checkpointing for HF models (if supported)
    def enable_gradient_checkpointing(self):
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                # Disable cache for gradient checkpointing compatibility
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = False
                self.model.gradient_checkpointing_enable()
                print("[INFO] Enabled gradient checkpointing for HF model; use_cache=False")
            elif hasattr(self.model, 'enable_input_require_grads'):
                # Some older APIs
                if hasattr(self.model, 'config'):
                    self.model.config.use_cache = False
                self.model.enable_input_require_grads()
                print("[INFO] Enabled gradient checkpointing (compat mode); use_cache=False")
        except Exception as e:
            print(f"[WARN] Failed to enable gradient checkpointing: {e}")
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embeddings if vocabulary changed.
        
        Args:
            new_vocab_size: New vocabulary size.
        """
        self.model.resize_token_embeddings(new_vocab_size)
        self.vocab_size = new_vocab_size


def load_hf_model(
    model_name: str = "gpt2",
    cache_dir: Optional[str] = None,
    pad_token_id: int = 50256
) -> HFModelWrapper:
    """
    Load a Hugging Face pretrained model.
    
    Supported models:
    - gpt2: GPT-2 small (117M parameters)
    - gpt2-medium: GPT-2 medium (345M parameters)
    - gpt2-large: GPT-2 large (774M parameters)
    - distilgpt2: DistilGPT-2 (82M parameters, faster)
    
    Args:
        model_name: Pretrained model name.
        cache_dir: Cache directory.
        pad_token_id: Padding token ID.
        
    Returns:
        Wrapped HF model.
    """
    return HFModelWrapper(
        model_name=model_name,
        cache_dir=cache_dir,
        pad_token_id=pad_token_id
    )


def get_model_from_config(config) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Model instance (either custom TransformerLM or HF wrapper).
    """
    import os
    from .transformer_lm import TransformerLM
    
    model_type = config.model.type.lower()
    
    if model_type == "custom":
        # Custom transformer model
        model = TransformerLM(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            max_seq_len=config.model.max_seq_len,
            dropout=config.model.dropout,
            gradient_checkpointing=getattr(config.training, 'gradient_checkpointing', False)
        )
        print(f"[INFO] Created custom TransformerLM:")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    elif model_type in ["gpt2", "distilgpt2", "gpt2-medium", "gpt2-large"]:
        # Pretrained HF model
        cache_dir = os.environ.get('HF_HOME', None)
        pretrained_name = config.model.get('pretrained_name', model_type)
        
        model = load_hf_model(
            model_name=pretrained_name,
            cache_dir=cache_dir
        )
        # Enable gradient checkpointing if requested
        if getattr(config.training, 'gradient_checkpointing', False):
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
                if hasattr(model.model, 'config'):
                    model.model.config.use_cache = False
                model.model.gradient_checkpointing_enable()
                print("[INFO] Enabled gradient checkpointing on inner HF model; use_cache=False")

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
