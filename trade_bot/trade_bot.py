import logging
import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from utils.log import prepare_logger

mpl.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "figure.figsize": (30, 10),
        "figure.dpi": 600,
    }
)
logger = prepare_logger(logging.INFO)


class BackTest:
    def __init__(
        self,
        transaction_cost: int,
        currency_count: int,
        ticker: str,
        scalers_path: Path,
        model_path: Path,
        asset_count: int = 0,
        verbose: bool = True,
        test_dates_path: Path = Path(__file__).parents[1]
        / "data"
        / "test_dates.csv",
        save_fig_path: Path = Path(__file__).parents[1] / "paper",
        save_plot: bool = False,
    ):
        self.currency_count = currency_count
        self.asset_count = asset_count
        self.current_sum_balance = currency_count
        self.transaction_cost = transaction_cost
        self.ticker = ticker
        self.shorted_assets_amount = 0
        self.short_start_price = 0
        self.model = self.load_model(model_path)
        self.X_scaler, self.y_scaler = self.load_scalers(scalers_path)
        self.current_asset_price = None
        self.initial_curr_count = currency_count
        self.test_dates = self.load_test_dates(test_dates_path)
        self.total_balance_history = self.get_initial_balance_history()
        self.total_balance_history_2 = self.get_initial_balance_history()
        self.save_fig_path = save_fig_path
        self.save_plot = save_plot

        if not verbose:
            logger.setLevel(logging.WARNING)

    def __repr__(self):
        return (
            f"Trade Bot - Ticker: {self.ticker} - Currency Balance: {self.current_sum_balance} "
            f"Asset Balance: {self.asset_count} - Total summed Balance: {self.current_sum_balance}"
            f"Shorted assets: {self.short_start_price}"
        )

    def load_model(self, load_path):
        return keras.models.load_model(load_path)

    def load_scalers(self, scalers_path):
        with open(scalers_path, "rb") as handle:
            scalers = pickle.load(handle)
        return scalers["X_scaler"], scalers["y_scaler"]

    def load_test_dates(self, test_dates_path):
        dates_df = pd.read_csv(test_dates_path)
        return pd.to_datetime(dates_df["dates"]).values

    def calculate_balance(self):
        return (
            self.currency_count + self.asset_count * self.current_asset_price
        )

    def predict_price(self, X):
        return self.model.predict(X)

    def get_initial_balance_history(self):
        return [self.initial_curr_count]

    def append_curr_balance_to_history(self, i):
        if i == 1:
            self.total_balance_history.append(self.calculate_balance())
        elif i == 2:
            self.total_balance_history_2.append(self.calculate_balance())
        else:
            raise ValueError("Wrong i value!")

    def calculate_asset_amount_by_price(
        self, price: float, curr_amount: float
    ):
        buy_amount = round(curr_amount / price)
        if buy_amount * price > self.currency_count:
            buy_amount -= 1
        if buy_amount >= 0:
            return buy_amount
        else:
            return 0

    def buy(self, amount: int, price: float):
        if not self.currency_count < price:
            self.currency_count -= amount * price + self.transaction_cost
            self.asset_count += amount
            logger.info(f"BUY {amount} - PRICE: {price}")
        else:
            logger.info("Abort! Not enough funds to buy stock!")

    def sell(self, amount: int, price: float):
        if self.asset_count != 0:
            self.currency_count += amount * price - self.transaction_cost
            self.asset_count -= amount
            logger.info(f"SELL {amount} - PRICE: {price}")
        else:
            logger.info("Abort! No assets to sell")

    def short(self, price):
        short_amount = self.calculate_asset_amount_by_price(
            price=price, curr_amount=self.currency_count
        )
        self.shorted_assets_amount = short_amount
        self.short_start_price = price
        logger.info(
            f"Short! - Amount: {short_amount} - Short start price {self.shorted_assets_amount}"
        )

    def rebuy_short(self, price):
        if self.shorted_assets_amount != 0:
            short_result = (
                self.short_start_price - price
            ) * self.shorted_assets_amount - self.transaction_cost
            self.currency_count += short_result
            logger.info(
                f"Realising short! - Result: {short_result} - Start price: {self.short_start_price} - End price: {price}"
            )
            self.short_start_price = 0
            self.shorted_assets_amount = 0

    def buy_and_hold(self, y):
        prices = [price[0] for price in y]
        amount = self.calculate_asset_amount_by_price(
            price=prices[0], curr_amount=self.initial_curr_count
        )
        left_cash = self.initial_curr_count - (prices[0] * amount)
        return [(price * amount) + left_cash for price in prices]

    def base_strategy(
        self,
        pred: float,
        last_price: float,
        top_cut_off: float,
        down_cut_off: float,
        if_short: bool,
    ):
        price_diff = pred - last_price
        if if_short:
            self.rebuy_short(last_price)
        logger.info(
            f"Predicted next price: {pred} - Last price: {last_price} - Diff: {price_diff}"
        )
        if price_diff > last_price * top_cut_off / 100:
            buy_amount = self.calculate_asset_amount_by_price(
                price=last_price, curr_amount=self.currency_count
            )
            self.buy(buy_amount, last_price)
        elif price_diff < -(last_price * down_cut_off / 100):
            sell_amout = self.asset_count
            self.sell(sell_amout, last_price)
            if if_short:
                self.short(price=last_price)
        else:
            logger.info("Price diffrence was not enough. Skipping trade")

    def inverse_scale(self, data: np.array, data_type: str):
        if data_type == "X":
            return self.X_scaler.inverse_transform(data)
        elif data_type == "y":
            return self.y_scaler.inverse_transform(data)
        else:
            raise ValueError(f'Wrong type: {data_type}. It must me "X" or "y"')

    def log_balances(self):
        logger.info(
            f"Asset amount: {self.asset_count} - Currency: {self.currency_count}"
            f"- Total Balance {self.calculate_balance()} - Asset Price: {self.current_asset_price}"
        )

    def simulate(
        self,
        X: np.array,
        y: np.array,
        top_cut_off: float,
        down_cut_off: float,
        if_short: bool,
        i: int,
    ):
        preds = self.inverse_scale(data=self.predict_price(X), data_type="y")
        y = self.inverse_scale(data=y, data_type="y")
        for pred, y_idx in zip(preds, range(len(y))):
            if y_idx in [0, len(y) - 1]:
                continue
            previous_close = y[y_idx - 1][0]
            self.current_asset_price = previous_close
            self.base_strategy(
                pred=pred[0],
                last_price=previous_close,
                top_cut_off=top_cut_off,
                down_cut_off=down_cut_off,
                if_short=if_short,
            )
            self.log_balances()
            self.append_curr_balance_to_history(i)

        logger.info(f"MAE: {mean_absolute_error(y, preds)}")
        self.current_asset_price = y[len(y) - 1][0]
        self.append_curr_balance_to_history(i)
        end_balance = self.calculate_balance()
        self.currency_count = self.initial_curr_count
        self.asset_count = 0
        logger.info(f"B&H Strategy result: {self.buy_and_hold(y)[-1]}")
        if i == 1:
            logger.info(
                f"Our Strategy result: {self.total_balance_history[-1]}"
            )
        elif i == 2:
            logger.info(
                f"Our Strategy result: {self.total_balance_history_2[-1]}"
            )
        return end_balance

    def simulate_2(
        self,
        X: np.array,
        y: np.array,
        top_cut_off: float,
        down_cut_off: float,
        if_short: bool,
        top_cut_off_2: float,
        down_cut_off_2: float,
        if_short_2: bool,
    ):
        end_balance_1 = self.simulate(
            X=X,
            y=y,
            top_cut_off=top_cut_off,
            down_cut_off=down_cut_off,
            if_short=if_short,
            i=1,
        )
        end_balance_2 = self.simulate(
            X=X,
            y=y,
            top_cut_off=top_cut_off_2,
            down_cut_off=down_cut_off_2,
            if_short=if_short_2,
            i=2,
        )
        if self.save_plot:
            return self.plot_results(y=y) # temp solution
        logger.info(
            f"END BALANCE1: {end_balance_1} - END BALANCE2: {end_balance_2}"
        )
        return end_balance_1, end_balance_2, self.buy_and_hold(y)[-1]

    @staticmethod
    def calculate_max_drawdown(portfolio_value):
        running_max = np.maximum.accumulate(portfolio_value)
        running_max[running_max < 1] = 1
        drawdown = (running_max - portfolio_value) / running_max
        return np.round(np.max(drawdown), 2)

    def plot_results(self, y):
        logger.info("Saving plot - start")
        buy_and_hold_results = self.buy_and_hold(y)
        plt.plot(
            self.test_dates,
            self.total_balance_history,
            color="blue",
            label=f"Proposed strategy with parameters found in training set hyperparameter optymalization.",
        )
        plt.plot(
            self.test_dates,
            self.total_balance_history_2,
            color="green",
            label=f"Proposed strategy with arbitrary chosen parameters.",
        )
        plt.plot(
            self.test_dates,
            buy_and_hold_results,
            color="red",
            label=f"Buy and Hold.",
        )
        plt.title(
            f"{self.ticker} proposed strategy with diffrent parameters vs Buy and Hold",
            size=30,
        )
        plt.xlabel("Date", size=25)
        plt.ylabel("Total Balance in USD", size=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.savefig(self.save_fig_path / f"{self.ticker}_strategy.pgf")
        logger.info("Saving plot - finished")
        return {
            "hyperopt_params": self.calculate_max_drawdown(self.total_balance_history),
            "arbitrary_params": self.calculate_max_drawdown(self.total_balance_history_2),
            "buy_and_hold": self.calculate_max_drawdown(buy_and_hold_results)
        } # temporary solution