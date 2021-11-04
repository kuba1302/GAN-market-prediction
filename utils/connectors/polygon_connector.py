import datetime
import urllib.request
import json

API_KEY = 'uAbu9bSmDF6XkqOuBawYuvdBspoueBys'


class PolygonConnector: 

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def get_one_day_data(self, ticker, date, adjusted=True):
        if adjusted:
            adjusted_str = 'true'

        else: 
            adjusted_str = 'false'

        url = f'https://api.polygon.io/v1/open-close/{ticker}/{date}?' \
        f'adjusted={adjusted_str}&apiKey={self.api_key}'
        response = urllib.request.urlopen(url)

        return json.loads(response.read())





