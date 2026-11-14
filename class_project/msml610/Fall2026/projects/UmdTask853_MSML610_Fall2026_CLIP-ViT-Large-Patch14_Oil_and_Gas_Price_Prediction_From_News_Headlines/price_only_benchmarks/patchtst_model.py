"""Compact, supervised PatchTST-style regressor for one-step *return* prediction.

This is a documented adaptation of PatchTST's key mechanisms (non-overlapping-
channel input, sliding temporal patches, positional embeddings, Transformer
encoder and flattened forecast head), NOT a verbatim implementation of Nie et
al.'s official benchmark, and not a pretrained foundation model.

The input sequence consists only of known past returns, ordered old -> recent.
Two optional auxiliary features (momentum, volatility) use the same past-only
information. The per-window normalization reads only those past returns.
"""
from __future__ import annotations

import torch
from torch import nn


class PatchTSTReturnRegressor(nn.Module):
    def __init__(self, *, context_length: int = 30, patch_len: int = 6,
                 stride: int = 3, hidden: int = 32, heads: int = 4,
                 layers: int = 2, dropout: float = 0.10):
        super().__init__()
        if not (context_length >= patch_len >= 2 and stride >= 1):
            raise ValueError('Invalid patch/context settings')
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.context_length = context_length
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (context_length - patch_len) // stride + 1
        self.patch_embedding = nn.Linear(patch_len, hidden)
        self.position_embedding = nn.Parameter(torch.empty(1, self.n_patches, hidden))
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * hidden + 2, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2 or seq.shape[1] != self.context_length:
            raise ValueError('seq must have shape (batch, context_length)')
        if aux.shape != (seq.shape[0], 2):
            raise ValueError('aux must have shape (batch, 2)')
        # Window-wise normalization uses past observations only.
        center = seq.mean(dim=1, keepdim=True)
        scale = seq.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        normalized = (seq - center) / scale
        patches = normalized.unfold(dimension=1, size=self.patch_len, step=self.stride)
        if patches.shape[1] != self.n_patches:
            raise AssertionError('Unexpected patch count')
        tokens = self.patch_embedding(patches) + self.position_embedding
        encoded = self.norm(self.encoder(tokens))
        features = torch.cat([encoded.flatten(start_dim=1), aux], dim=1)
        return self.head(features).squeeze(-1)
