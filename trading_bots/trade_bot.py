import pandas as pd 
import numpy as np 



class TradeBot: 

    def __init__(self, transaction_cost: int, currency_count: int, asset_count: int = 0):
        self.currency_count = currency_count
        self.asset_count = asset_count
        # at the beggining balance is our currency amount 
        self.current_balance = currency_count
        self.current_price = None 
        self.transaction_cost = transaction_cost
