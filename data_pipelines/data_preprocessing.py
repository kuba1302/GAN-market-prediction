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
        if_ta=True,
    ) -> None:
        self.ticker = ticker
        self.step_train = step_train
        self.step_predict = step_predict
        self.data_path = data_path
        self.save_path = save_path
        self.sentiment_path = sentiment_path
        self.if_ta = if_ta
        self.min_time = None
        self.max_time = None
        self.sentiment_df = self.load_sentiment_data()

    def load_sentiment_data(self):
        sentiment_df = pd.read_csv(self.sentiment_path / f"{self.ticker}.csv").drop(
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

    def _load_stock_data(self):
        stock_data = pd.read_csv(self.data_path / f"{self.ticker}.csv")
        stock_data.set_index(pd.DatetimeIndex(stock_data["Date"]))
        return stock_data

    def prepare_sentiment_data(self):
        AGGREGATIONS = {
            "sentiment": ["count", "mean", "std", "median", q_at(0.25), q_at(0.75)]
        }
        sentiment_df = self.load_sentiment_data()
        sentiment_df["sentiment"] = sentiment_df["sentiment"].astype(float)
        return sentiment_df.groupby("Date").agg(AGGREGATIONS)["sentiment"]

    def prepare_data(
        self,
        X: pd.DataFrame,
        X_cols: list,
        save_scaler_path: Path,
        y_col: str = "Close",
    ):
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        print(X.loc[:, X_cols])
        X_df = pd.DataFrame(X_scaler.fit_transform(X.loc[:, X_cols]), columns=X_cols)
        y_series = y_scaler.fit_transform(X.loc[:, y_col].values.reshape(-1, 1))
        X_df[y_col] = y_series
        scalers = {"X_scaler": X_scaler, "y_scaler": y_scaler}
        with open(save_scaler_path / f"scalers_{self.ticker}.pickle", "wb") as handle:
            pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return X_df

    def split_data(
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

    def train_test_split(self, data, train_percent):
        split_idx = round(len(data) * train_percent)
        return data[:split_idx], data[split_idx:]

    def prepare_final_data(self, sentiment=None):
        stock_data = self._load_stock_data()
        if self.if_ta:
            stock_data = self.perform_technical_analysis(stock_data)
        final_data = (
            stock_data.loc[
                (stock_data["Date"] >= self.min_time)
                & (stock_data["Date"] < self.max_time)
            ]
            .drop(
                columns=["Dividends", "Stock Splits", "Unnamed: 0"]
                if "Unnamed: 0" in stock_data
                else ["Dividends", "Stock Splits"]
            )
            .dropna(axis="columns")
        )
        if sentiment is not None:
            final_data = (
                final_data.merge(self.sentiment_df, how="left", on="Date")
                .set_index(pd.DatetimeIndex(final_data["Date"]))
                .drop(columns="Date")
                .interpolate(method="time")
            )
        x_cols = [col for col in final_data.columns if col not in ["Close", "Date"]]
        scaled_data = self.prepare_data(
            X=final_data, X_cols=x_cols, save_scaler_path=self.save_path
        )
        X_list, Y_preds_real_list, Y_whole_real_list = self.split_data(
            scaled_data, self.step_train, self.step_predict
        )
        return X_list, Y_preds_real_list, Y_whole_real_list

    def save_data(self, sentiment=None):
        X_list, Y_preds_real_list, Y_whole_real_list = self.prepare_final_data(
            sentiment=sentiment,
        )
        df_lists = [X_list, Y_preds_real_list, Y_whole_real_list]
        names = ["X_list", "Y_preds_real_list", "Y_whole_real_list"]
        temp = dict(zip(names, df_lists))
        save_data = {}
        for name, df_list in temp.items():
            train, test = self.train_test_split(df_list, 0.75)
            save_data[f"{name}_train"] = train
            save_data[f"{name}_test"] = test

        with open(self.save_path / f"data_{self.ticker}.pickle", "wb") as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_moving_averages(self, X, peroid_list=[5, 10, 20, 50, 100]):
        df = X.copy()
        for peroid in peroid_list:
            df[f"sma_{peroid}"] = df["Close"].rolling(window=peroid).mean()
            df[f"ema_{peroid}"] = df["Close"].ewm(span=peroid).mean()

            weights = np.arange(1, peroid + 1)
            df[f"wma_{peroid}"] = (
                df["Close"]
                .rolling(window=peroid)
                .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
            )

            bollinger_up, bollinger_down = self.get_bollinger_bands(df["Close"], peroid)
            df[f"bb_{peroid}_up"] = bollinger_up
            df[f"bb_{peroid}_down"] = bollinger_down
        return df

    @staticmethod
    def get_bollinger_bands(prices, peroid):
        sma = prices.rolling(window=peroid).mean()
        std = prices.rolling(peroid).std()
        bollinger_up = sma + std * 2
        bollinger_down = sma - std * 2
        return bollinger_up, bollinger_down

    @staticmethod
    def get_macd(prices):
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        return macd, signal_line

    def perform_technical_analysis(self, X, peroid_list=[5, 10, 20, 50, 100]):
        df = X.copy()
        df = self.get_moving_averages(X, peroid_list)
        df["macd"], df["signal_line"] = self.get_macd(df["Close"])
        return df
