import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trade_bot import BackTest

if __name__ == "__main__":
    models_dict =  {
    "UBSFY": {"version": 0.1, 
              "top_cut_off": 1.8,
              "down_cut_off": 1.4
              },
    "EA": {"version": 0.6, 
              "top_cut_off": 0.4,
              "down_cut_off": 0.6
              },
   "TTWO": {"version": 0.1, 
              "top_cut_off": 1.6,
              "down_cut_off": 0.4
              },
   "ATVI": {"version": 0.1, 
              "top_cut_off": 0.0,
              "down_cut_off": 2.2
              },
    }
    
    drop_backs = {}
    
    for ticker, info in models_dict.items():
        scaled_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
        scalers_path = scaled_path / f"scalers_{ticker}.pickle"
        data_path = scaled_path / f"data_{ticker}.pickle"
        version = info['version']
        top_cut_off = info['top_cut_off']
        down_cut_off = info["down_cut_off"]
        model_path = (
            Path(os.path.abspath("")).parents[0]
            / "models"
            / "gan"
            / "versions"
            / f"model_{version}_{ticker}class"
        )

        bot = BackTest(
            transaction_cost=0.0007,
            currency_count=1000,
            ticker=ticker,
            scalers_path=scalers_path,
            model_path=model_path,
            verbose=True,
            save_plot=True,
        )
        with open(data_path, "rb") as handle:
            data = pickle.load(handle)

        X = data["X_list_test"]
        y = data["Y_preds_real_list_test"]
        print(y)
        drop_backs[ticker] = bot.simulate_2(
            X=X,
            y=y,
            top_cut_off=top_cut_off,
            down_cut_off=down_cut_off,
            if_short=True,
            top_cut_off_2=2,
            down_cut_off_2=0,
            if_short_2=True,
        )
        print(drop_backs)