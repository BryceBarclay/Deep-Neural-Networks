# classic libraries
import numpy as np

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


# function assigning labels based on max values of the next nbars_crit candles
def get_label(close, max_list, pts):
    if close + pts < np.amax(max_list):
        label = 1
        # print(close, np.amax(max_list))
    else:
        label = 0
    return label

