# lstm_model.py
# This file trains an LSTM model that uses the past sentiment values
# to predict if the stock price will go up or down next.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from config import WINDOW_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE


# make sliding windows of sentiment values
def create_sequences(X, y, window):
    X_seq, y_seq = [], []
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(merged_df):
    # target variable
    merged_df["future_return"] = merged_df["Close"].pct_change().shift(-1)
    merged_df = merged_df.dropna()

    # only sentiment is used as input for now
    X = merged_df[["sentiment"]].values
    y = (merged_df["future_return"] > 0).astype(int).values   

    # scale sentiment values between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # make time windows 
    X_seq, y_seq = create_sequences(X_scaled, y, WINDOW_SIZE)

    # train/test split 
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # LSTM model
    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(WINDOW_SIZE, 1)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the LSTM
    model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=1)

    # test accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nLSTM Model Accuracy: {acc}\n")

    return model, scaler
