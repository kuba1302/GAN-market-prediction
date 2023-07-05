from gc import callbacks


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense


def lstm_nn(input_dim, feature_size, output_dim=1, optimizer="Adam", loss="rmse"):
    model = Sequential()
    model.add(
        LSTM(
            units=512,
            return_sequences=True,
            input_shape=(input_dim, feature_size),
            recurrent_dropout=0,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=False,
            use_bias=True,
            kernel_regularizer="l2",
            name="LSTM1",
        )
    )
    model.add(
        LSTM(
            units=256,
            return_sequences=True,
            recurrent_dropout=0,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=False,
            use_bias=True,
            kernel_regularizer="l2",
            name="LSTM2",
        )
    )
    model.add(
        LSTM(
            units=128,
            return_sequences=False,
            recurrent_dropout=0,
            activation="tanh",
            recurrent_activation="sigmoid",
            unroll=False,
            use_bias=True,
            kernel_regularizer="l2",
            name="LSTM3",
        )
    )
    model.add(Dense(units=output_dim))
    model.compile(optimizer=optimizer, metrics=["mae"], loss=loss)
    return model