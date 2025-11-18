"""
Custom Transformer Language Model (Decoder-only, GPT-like).

Implements a causal transformer for autoregressive text generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Attention mask of shape (batch, 1, seq_len, seq_len).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask (causal mask for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension.
            d_ff: Hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with GELU activation."""
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.
        
        Args:
            x: Input tensor.
            mask: Attention mask.
            
        Returns:
            Output tensor.
        """
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model (decoder-only, GPT-like).
    
    This is a causal transformer for autoregressive text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 50256,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize Transformer Language Model.
        
        Args:
            vocab_size: Size of vocabulary.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward hidden dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            pad_token_id: Padding token ID.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.gradient_checkpointing = gradient_checkpointing
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection (weight tying)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            seq_len: Sequence length.
            device: Device to create mask on.
            
        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len).
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len). 1 for real tokens, 0 for padding.
            labels: Target labels for loss computation (batch, seq_len).
            
        Returns:
            Tuple of (logits, loss).
            - logits: Output logits of shape (batch, seq_len, vocab_size).
            - loss: Cross-entropy loss if labels provided, else None.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask (1, 1, seq_len, seq_len)
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # Convert attention_mask to (batch, 1, seq_len, seq_len) for proper masking
            # attention_mask: (batch, seq_len) where 1 = real token, 0 = padding
            # We need to mask out positions where either:
            #   1. Causal: future tokens (already in causal_mask)
            #   2. Padding: padding tokens (from attention_mask)
            # Create key mask: (batch, 1, 1, seq_len) - which positions can be attended to
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # Create query mask: (batch, 1, seq_len, 1) - which positions are queries
            query_mask = attention_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, seq_len, 1)
            # Combine: (batch, 1, seq_len, seq_len)
            # Both query and key must be valid (non-padding), AND causal constraint
            padding_mask = key_mask * query_mask  # (batch, 1, seq_len, seq_len)
            # Combine with causal mask: both must be 1
            combined_mask = causal_mask * padding_mask  # (batch, 1, seq_len, seq_len)
        else:
            combined_mask = causal_mask
        
        # Pass through transformer blocks
        if self.gradient_checkpointing:
            # Use PyTorch activation checkpointing to save memory
            from torch.utils.checkpoint import checkpoint

            def _block_fn(b, x_in, m_in):
                return b(x_in, m_in)

            for block in self.blocks:
                x = checkpoint(lambda xi, mi: _block_fn(block, xi, mi), x, combined_mask)
        else:
            for block in self.blocks:
                x = block(x, combined_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten for cross-entropy
            # Use ignore_index=-100 to ignore padding (allows real EOS tokens to contribute to loss)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token IDs of shape (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-k sampling (keep only top k tokens).
            top_p: Nucleus sampling (keep tokens with cumulative prob >= top_p).
            eos_token_id: End-of-sequence token ID to stop generation.
            
        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens).
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (use only last max_seq_len tokens to avoid OOM)
                current_input = input_ids[:, -self.max_seq_len:]
                logits, _ = self.forward(current_input)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token generated
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return input_ids
