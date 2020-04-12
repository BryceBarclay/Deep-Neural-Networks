# get data libraries

import yfinance as yf
import pandas_datareader.data as pdr
yf.pdr_override()

def get_stocks(start, end, stock):
    """
    Gets stock data from Yahoo finance
    """
    data = pdr.get_data_yahoo(stock, start, end)
    stocks = data["Adj Close"]
    stocks.to_csv("../dataset/stocks.csv")


