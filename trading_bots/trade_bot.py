import numpy as np
import os
import pickle
from utils.log import prepare_logger
import logging
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import mean_absolute_error

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

    def calculate_balance(self):
        return self.currency_count + self.asset_count * self.current_asset_price

    def predict_price(self, X):
        return self.model.predict(X)

    def calculate_asset_amount_by_price(self, price: float, curr_amount: float):
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
        logger.info(f"Short! - Amount: {short_amount} - Short start price {self.shorted_assets_amount}")

    def rebuy_short(self, price):
        if self.shorted_assets_amount != 0:
            short_result = (self.short_start_price - price) * self.shorted_assets_amount - self.transaction_cost
            self.currency_count += short_result
            logger.info(f"Realising short! - Result: {short_result} - Start price: {self.short_start_price} - End price: {price}")
            self.short_start_price = 0
            self.shorted_assets_amount = 0

    def base_strategy(
        self, pred: float, last_price: float, top_cut_off: float, down_cut_off: float, if_short: bool
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
        self, X: np.array, y: np.array, top_cut_off: float, down_cut_off: float, if_short: bool
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
                if_short=if_short
            )
            self.log_balances()
        logger.info(f'MAE: {mean_absolute_error(y, preds)}')
