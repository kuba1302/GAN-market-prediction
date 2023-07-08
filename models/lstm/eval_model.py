import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
import pickle
from pathlib import Path
import os
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def eval_model(ticker, version):
    models_path = Path("/home/ubuntu/projects/GAN-market-prediction/models/lstm/versions")
    load_path = Path("/home/ubuntu/projects/GAN-market-prediction/data/scaled_data")

    with open(load_path / f"data_{ticker}.pickle", "rb") as handle:
        data = pickle.load(handle)

    with open(load_path / f"scalers_{ticker}.pickle", "rb") as handle:
        scalers = pickle.load(handle)
        
    model = keras.models.load_model(models_path / f"lstm_{ticker}_{version}")
    preds = model.predict(data["X_list_test"])
    backtest_path = Path(os.path.abspath("")).parents[1] / "backtesting"
    actual_values = data["Y_preds_real_list_test"]
    test_preds = scalers["y_scaler"].inverse_transform(preds)
    test_true = scalers["y_scaler"].inverse_transform(actual_values)
    return {
    "mae": mean_absolute_error(y_true=test_true, y_pred=test_preds),
    "mape": mean_absolute_percentage_error(y_true=test_true, y_pred=test_preds),
    "rmse": mean_squared_error(y_true=test_true, y_pred=test_preds) ** 0.5
    }


def eval_all_models_on_version(version):
    tickers = ["EA", "UBSFY", "ATVI", "TTWO"]
    model_evals = {}
    for ticker in tickers:
        model_evals[ticker] = eval_model(ticker, version)
    
    return model_evals

def main():
    versions = ["0.1_PURE_DATA"]
    results = {}
    for version in versions:
        results[version] = eval_all_models_on_version(version)
        
    return results

if __name__ == "__main__":
    print(main())