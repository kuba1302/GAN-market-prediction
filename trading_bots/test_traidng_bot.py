import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot import BackTest

if __name__ == "__main__":
    TICKER = "UBSFY"
    MODEL_VERSION = "0.2"

    scaled_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    scalers_path = scaled_path / f"scalers_{TICKER}.pickle"
    data_path = scaled_path / f"data_{TICKER}.pickle"
    model_path = (
        Path(os.path.abspath("")).parents[0]
        / "models"
        / "gan"
        / "versions"
        / f"model_{MODEL_VERSION}_{TICKER}class"
    )

    bot = BackTest(
        transaction_cost=0.0007,
        currency_count=1000,
        ticker="EA",
        scalers_path=scalers_path,
        model_path=model_path,
    )
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    X = data["X_list_test"]
    y = data["Y_preds_real_list_test"]

    bot.simulate(X=X, y=y, top_cut_off=2, down_cut_off=0, if_short=True)
