# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
# classic libraries
import numpy as np
#import statsmodels.api as sm    # to estimate an average with 'loess'
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import random, string
import os, time, datetime, json
# personal libraries
from utils.toolbox import *
from utils.all_models import *

#-------------------------------------------------#
#               A) Hyper-parameters               #
#-------------------------------------------------#
hyperP = {
    'data_to_read': '../dataset/AMZN.csv',
    'model': 'LSTM_RNN',  # 'LSTM_RNN', 'GRU_RNN' 
    'hidden_size': 40,
    'n_steps' : 2001,
    'lr' : .005,
    'sequence_len' : 100,   # x_1 x_2 ... x_100
    'print_every' : 200,
    'compute_perplexity_every' : 1000,
    'folder_result': '../results'
}

# using yfinance package:
#start = '2010-01-01'
#end   = '2011-01-01'
#stock = 'APPL'
#get_stocks(start, end, stock)


# using manually downloaded data:
myStocks = pd.read_csv(hyperP['data_to_read'])
print(myStocks['Adj Close'])



