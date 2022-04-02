import pandas as pd 
import numpy as np 
import dill 


class TradeBot: 

    def __init__(self, transaction_cost: int, currency_count: int, ticker: str, asset_count: int = 0):
        self.currency_count = currency_count
        self.asset_count = asset_count
        # at the beggining balance is our currency amount 
        self.current_balance = currency_count
        self.transaction_cost = transaction_cost
        self.ticker = ticker
        self.current_price = None 
        self.model = None 

    def __repr__(self):
        return f'Trade Bot - Ticker: {self.ticker} - Balance: {self.current_balance}'

    def load_model(self, path): 
        with open(path, 'r') as handle: 
            self.model = dill.load(handle)
    
    