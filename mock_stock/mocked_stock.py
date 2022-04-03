import pandas as pd 
from pathlib import Path


class MockedStock: 
    """
    might be usefull later, for now every simulation is performed 
    in trading_bots/
    """
    def __init__(self, data_path: Path, ticker: str):
        self.data_path = data_path
        self.ticker = ticker
        self.data = self.load_data(data_path)

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, position: int): 
        return self.data.iloc[position].to_dict()

    def __repr__(self): 
        return f"Mocked Stock - Ticker: {self.ticker}"
    
    def __iter__(self): 
        for day_info in self.data.itertuples():
            yield day_info._asdict()

    def load_data(self): 
        return pd.read_csv(self.data_path)
    
    