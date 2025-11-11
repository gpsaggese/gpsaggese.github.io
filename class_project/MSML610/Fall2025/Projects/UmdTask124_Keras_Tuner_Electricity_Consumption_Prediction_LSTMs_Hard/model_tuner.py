import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_tunable(hp, input_shape):
    model = Sequential()
    units = hp.Int('lstm_units', 32, 128, step=32)
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units', 16, 128, step=16), activation='relu'))
    model.add(Dense(1))
    lr = hp.Choice('learning_rate', [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model
