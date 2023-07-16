import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import os
import pickle
from pathlib import Path

import numpy as np
from model import lstm_nn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization


def train_model(ticker, version):
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    EPOCHS = 100
    load_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    save_path = (
        Path(os.path.abspath("")).parents[0] / "models" / "lstm" / "versions"
    )
    os.makedirs(save_path, exist_ok=True)

    with open(load_path / f"data_{ticker}.pickle", "rb") as test:
        data = pickle.load(test)
    print(
        "-----------------------------------------------"
        f'TRAIN DATA SHAPE: {data["X_list_train"].shape}'
        " ----------------------------------------------"
    )
    lstm = lstm_nn(
        input_dim=data["X_list_train"].shape[1],
        feature_size=data["X_list_train"].shape[2],
        optimizer="Adam",
        loss="mse",
    )
    mc = ModelCheckpoint(
        save_path / f"lstm_{version}",
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=1, patience=20
    )
    lstm.fit(
        data["X_list_train"],
        data["Y_preds_real_list_train"],
        epochs=EPOCHS,
        callbacks=[early_stopping],
        batch_size=8,
    )
    lstm.save(save_path / f"lstm_{ticker}_{version}")


if __name__ == "__main__":
    MODEL_VERSION = "0.1_TA_SENTIMENT"
    tickers = ["EA", "UBSFY", "ATVI", "TTWO"]
    for ticker in tickers:
        print(
            "-----------------------------------------------"
            f"TRAINING MODEL: {ticker} - {MODEL_VERSION}"
            " ----------------------------------------------"
        )
        train_model(ticker, MODEL_VERSION)
