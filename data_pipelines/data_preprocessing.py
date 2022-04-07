import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.quantiles import q_at

# data_path = Path(os.path.abspath("")) / "data"
# sentiment_path = data_path / "sentiment"
# save_path = Path(os.path.abspath('')) / 'data' / 'scaled_data' 

class DataPipeline:
    def __init__(
        self,
        ticker: str,
        data_path: Path,
        save_path: Path, 
        sentiment_path: Path,
        step_train: int,
        step_predict: int,
    ) -> None:
        self.ticker = ticker
        self.step_train = step_train
        self.step_predict = step_predict
        self.min_time = None
        self.max_time = None
        self.sentiment_df = self.prepare_sentiment_data(sentiment_path=sentiment_path)

    def load_sentiment_data(self, sentiment_path: Path):
        sentiment_df = pd.read_csv(sentiment_path / f"{self.ticker}.csv").drop(
            columns=["Unnamed: 0", "Unnamed: 0.1"]
        )
        sentiment_df = sentiment_df[sentiment_df["created_utc"] != "created_utc"]
        sentiment_df["Date"] = (
            sentiment_df["created_utc"]
            .astype(int)
            .apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%d-%m"))
        )
        self.min_time = sentiment_df["Date"].min()
        self.max_time = sentiment_df["Date"].max()
        return sentiment_df

    def _load_stock_data(self, data_path, ticker):
        stock_data = pd.read_csv(data_path / f"{ticker}.csv")
        stock_data.set_index(pd.DatetimeIndex(stock_data['Date']))
        return stock_data

    def prepare_sentiment_data(self, sentiment_path: Path):
        AGGREGATIONS = {
            "sentiment": ["count", "mean", "std", "median", q_at(0.25), q_at(0.75)]
        }
        sentiment_df = self.read_sentiment_data(sentiment_path=sentiment_path)
        sentiment_df["sentiment"] = sentiment_df["sentiment"].astype(float)
        return sentiment_df.groupby("Date").agg(AGGREGATIONS)["sentiment"]

    def _prepare_data(
        self,
        X: pd.DataFrame,
        X_cols: list,
        save_scaler_path: Path,
        y_col: str = "Close",
    ):
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_df = pd.DataFrame(X_scaler.fit_transform(X.loc[:, X_cols]), columns=X_cols)
        y_series = y_scaler.fit_transform(X.loc[:, y_col].values.reshape(-1, 1))
        X_df[y_col] = y_series
        scalers = {"X_scaler": X_scaler, "y_scaler": y_scaler}
        with open(save_scaler_path / f"scalers{self.ticker}.pickle", "wb") as handle:
            pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return X_df

    def _split_data(
        self, X: pd.DataFrame, step_train: int, step_predict: int, y_col: str = "Close"
    ):
        data_len = X.shape[0]
        X_list = []
        Y_preds_real_list = []
        Y_whole_real_list = []
        for i in range(data_len):
            X_step = X.loc[
                i : i + step_train - 1, [col for col in X.columns if col != y_col]
            ]
            Y_pred_real = X.loc[
                i + step_train : i + step_train + step_predict - 1, y_col
            ]
            Y_whole_real = X.loc[i : i + step_train - 1, y_col]
            if (len(X_step) == step_train) & (len(Y_pred_real) == step_predict):
                X_list.append(X_step)
                Y_preds_real_list.append(Y_pred_real)
                Y_whole_real_list.append(Y_whole_real)
        return (
            np.array(X_list),
            np.array(Y_preds_real_list),
            np.array(Y_whole_real_list),
        )

    def train_test_split(data, train_percent):
        split_idx = round(len(data) * train_percent)
        return data[:split_idx], data[split_idx:]


    def prepare_final_data(self, sentiment=None):
        final_data = (
            stock_data.loc[(stock_data["Date"] >= min_time) & (stock_data["Date"] < max_time)]
            .drop(columns=["Dividends", "Stock Splits", "Unnamed: 0"])
            .dropna(axis="columns")
        )
        if sentiment is not None: 
            sentiment_df = self.prepare_sentiment_statistics(sentiment)
            final_data = (final_data.merge(sentiment_df, how='left', on='Date')
                                    .set_index(pd.DatetimeIndex(final_data['Date']))
                                    .drop(columns='Date')
                                    .interpolate(method='time'))
        x_cols = [col for col in final_data.columns if col not in ["Close", "Date"]]
        scaled_data = self.prepare_data(X=final_data, X_cols=x_cols, save_scaler_path=self.save_path)
        X_list, Y_preds_real_list, Y_whole_real_list = split_data(
            scaled_data, STEP_TRAIN, STEP_PREDICT
        )
        return X_list, Y_preds_real_list, Y_whole_real_list