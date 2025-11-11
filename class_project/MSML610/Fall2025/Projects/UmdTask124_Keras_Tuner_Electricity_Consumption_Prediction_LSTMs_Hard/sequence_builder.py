import numpy as np


def create_sequences(features, target, seq_len=24, target_step=1):
    X, y = [], []
    for i in range(len(features) - seq_len - target_step + 1):
        X.append(features[i:i+seq_len])
        y.append(target[i+seq_len+target_step-1])
    return np.array(X), np.array(y)


def time_split(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])
