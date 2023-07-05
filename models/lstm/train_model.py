import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import os
import numpy as np
import pickle
import os
from pathlib import Path
from model import lstm_nn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization


if __name__ == "__main__":
    # from tensorflow.python.client import device_lib 
    # print(device_lib.list_local_devices())
    MODEL_VERSION = "0.1"
    EPOCHS = 100
    TICKER = "UBSFY"
    load_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    save_path = Path(os.path.abspath("")).parents[0] / "models" / "lstm" / "versions"
    os.makedirs(save_path, exist_ok=True)
    
    with open(load_path / f"data_{TICKER}.pickle", "rb") as test:
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
        save_path / f"lstm_{MODEL_VERSION}",
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=20)
    lstm.fit(
        data["X_list_train"],
        data["Y_preds_real_list_train"],
        epochs=EPOCHS,
        callbacks=[early_stopping],
        batch_size=8
    )
    lstm.save(save_path / f"lstm_{TICKER}_{MODEL_VERSION}")