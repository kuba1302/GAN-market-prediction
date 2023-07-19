import itertools
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from utils.log import prepare_logger

from trade_bot import BackTest

logger = prepare_logger(logging.INFO)


hparams = {
    "top_cut_off": np.arange(0, 4, 0.2),
    "down_cut_off": np.arange(0, 4, 0.2),
    "if_short": [True, False],
}

companies_dict = {
    # "UBSFY": 0.1,
    "EA": 0.6,
    # "TTWO": 0.1,
    # "ATVI": 0.1
}


def get_all_permutations(hparams: dict):
    keys, values = zip(*hparams.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations


def hyperopt(ticker, model_version):
    scaled_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
    scalers_path = scaled_path / f"scalers_{ticker}.pickle"
    data_path = scaled_path / f"data_{ticker}.pickle"
    save_path = (
        Path(os.path.abspath("")) / f"{ticker}_{model_version}_hyperopt"
    )
    model_path = (
        Path(os.path.abspath("")).parents[0]
        / "models"
        / "gan"
        / "versions"
        / f"model_{model_version}_{ticker}class"
    )

    bot = BackTest(
        transaction_cost=0.0007,
        currency_count=1000,
        ticker="EA",
        scalers_path=scalers_path,
        model_path=model_path,
        verbose=False,
    )
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    X = data["X_list_train"]
    y = data["Y_preds_real_list_train"]

    results = pd.DataFrame()

    hyper_permutations = get_all_permutations(hparams)
    print(f"NUMBER OF PERMUTATIONS: {len(hyper_permutations)}")
    for hyper_permutation in hyper_permutations:
        tmp = bot.simulate(X=X, y=y, **hyper_permutation, i=1)
        print(f"TICKER: {ticker} - {hyper_permutation} - END_BALANCE: {tmp}")
        results = results.append({"end_balance": tmp}, ignore_index=True)

    results = pd.concat(
        [results, pd.Series(hyper_permutations).apply(pd.Series)], axis=1
    )
    results.to_csv(save_path / f"{save_path}.csv", index=False)


def multiple_hyper_opt(companies_dict):
    for ticker, model_version in companies_dict.items():
        hyperopt(ticker=ticker, model_version=model_version)


if __name__ == "__main__":
    multiple_hyper_opt(companies_dict)
