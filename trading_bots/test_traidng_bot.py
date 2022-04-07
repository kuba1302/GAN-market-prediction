import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot import BackTest

if __name__ == "__main__":
    MODEL_VERSION = "0.6"

    scaled_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    scalers_path = scaled_path / "scalers.pickle"
    data_path = scaled_path / "data.pickle"
    model_path = (
        Path(os.path.abspath("")).parents[0]
        / "models"
        / "gan"
        / "versions"
        / f"model_{MODEL_VERSION}_class"
    )

    bot = BackTest(
        transaction_cost=0.0007,
        currency_count=1000,
        ticker="EA",
        scalers_path=scalers_path,
        model_path=model_path,
    )
    with open(scaled_path / "data.pickle", "rb") as handle:
        data = pickle.load(handle)

    X = data["X_list_train"]
    y = data["Y_preds_real_list_train"]

    bot.simulate(X, y, 0, 0)
