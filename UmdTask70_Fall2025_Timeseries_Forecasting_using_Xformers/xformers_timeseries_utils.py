import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional

# --- ROBUST IMPORTS ---
# Try importing xformers components. If they are missing or referenced differently,
# we fall back to standard PyTorch components to prevent a crash.
try:
    import xformers
    import xformers.ops
    # 1. Try importing high-level components (newer xformers)
    try:
        from xformers.components import MultiHeadDispatch
        from xformers.components.attention import ScaledDotProduct
    except ImportError:
        # 2. If 'components' namespace is missing, check strict warning or fallback
        print("Warning: xformers.components not found. Instantiating fallback classes.")
        MultiHeadDispatch = None
        ScaledDotProduct = None

except ImportError:
    print("Warning: xformers library not found. Using standard PyTorch fallback.")
    MultiHeadDispatch = None
    ScaledDotProduct = None

# Fallback implementations (Mock classes wrapping standard PyTorch)
# This guarantees 'MultiHeadDispatch' and 'ScaledDotProduct' always exist in namespace.
if MultiHeadDispatch is None:
    class ScaledDotProduct(nn.Module):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.dropout = dropout
        def forward(self, q, k, v, att_mask=None, key_padding_mask=None):
            # Using PyTorch 2.0+ scaled_dot_product_attention if available, else manual
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                 return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=att_mask, dropout_p=self.dropout)
            else:
                 # Minimal manual implementation
                 dk = q.size(-1)
                 scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
                 if att_mask is not None:
                     scores = scores.masked_fill(att_mask == 0, -1e9)
                 attn = torch.functional.F.softmax(scores, dim=-1)
                 return torch.matmul(attn, v)

    class MultiHeadDispatch(nn.Module):
        def __init__(self, dim_model, num_heads, attention=None, use_rotary_embeddings=False):
            super().__init__()
            # Wrapper around standard MultiheadAttention
            self.mha = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)
        
        def forward(self, query, key, value, **kwargs):
            # xformers arguments might vary, but for this project we generally pass q,k,v
            # Output of mha is (attn_output, attn_output_weights)
            return self.mha(query, key, value)[0]

print(f"Xformers components initialized. Using fallback: {MultiHeadDispatch.__name__ == 'MultiHeadDispatch' and MultiHeadDispatch.__module__ == __name__}")

# --- DATASET ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, target_len: int = 1):
        """
        Args:
            data (np.ndarray): The time series data (num_samples, num_features).
            seq_len (int): Length of the input sequence.
            target_len (int): Length of the target sequence (prediction horizon).
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_len + 1

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.seq_len]
        tgt = self.data[idx + self.seq_len : idx + self.seq_len + self.target_len]
        return src, tgt

# --- MODEL ---
class XformersTimeSeriesModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model)) 
        
        # Use the components defined above (either real xformers or fallback)
        self.attn = MultiHeadDispatch(
            dim_model=d_model,
            num_heads=nhead,
            attention=ScaledDotProduct(), 
            use_rotary_embeddings=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.num_layers = num_layers
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        x = self.input_projection(src)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        for _ in range(self.num_layers):
            attn_out = self.attn(x, x, x)
            x = self.norm1(x + self.dropout1(attn_out))
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout2(ff_out))
        
        last_output = x[:, -1, :]
        prediction = self.decoder(last_output)
        return prediction

# --- PREPROCESSOR ---
class DataPreprocessor:
    def __init__(self, seq_len: int, train_split: float = 0.8):
        self.scaler = MinMaxScaler()
        self.seq_len = seq_len
        self.train_split = train_split

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
        data = df[feature_cols].values
        split_idx = int(len(data) * self.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        train_scaled = self.scaler.fit_transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        train_dataset = TimeSeriesDataset(train_scaled, self.seq_len)
        test_dataset = TimeSeriesDataset(test_scaled, self.seq_len)
        
        return train_dataset, test_dataset, self.scaler

# --- TRAIN & EVAL ---
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            target = tgt[:, 0, 0].unsqueeze(1) 
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
    return loss_history

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            target = tgt[:, 0, 0].unsqueeze(1)
            output = model(src)
            predictions.append(output.cpu().numpy())
            actuals.append(target.cpu().numpy())
            
    return np.vstack(predictions), np.vstack(actuals)
