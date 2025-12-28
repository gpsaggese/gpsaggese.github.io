"""
sequence_model.py

Functions to create sequences and build a simple LSTM model.
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(features, target, seq_len=24, target_step=1):
    X, y = [], []
    total = len(features)
    end = total - seq_len - target_step + 1
    for i in range(end):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len+target_step-1])
    return np.array(X), np.array(y)

def build_baseline(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
