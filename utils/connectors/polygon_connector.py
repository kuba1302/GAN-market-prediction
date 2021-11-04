import datetime
import typing_extensions
import urllib.request
import json

API_KEY = 'uAbu9bSmDF6XkqOuBawYuvdBspoueBys'


def get_one_day_data(stocksTicker, date, api_key, adjusted=True):
    if adjusted:
        adjusted_str = 'true'
    else: 
        adjusted_str = 'false'

    url = f'https://api.polygon.io/v1/open-close/AAPL/{date}?' \
    f'adjusted={adjusted_str}&apiKey={api_key}'
    print(url)
    response = urllib.request.urlopen(url)

    return json.loads(response.read())


data = get_one_day_data('ATVI', '2021-11-02', api_key=API_KEY)
print(data)


