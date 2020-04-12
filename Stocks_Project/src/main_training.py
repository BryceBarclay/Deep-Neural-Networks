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
    'hidden_size': 10,
    'n_steps' : 2001,
    'lr' : .005,
    'sequence_len' : 100,   # x_1 x_2 ... x_100
    'print_every' : 200,
    'compute_perplexity_every' : 1000,
    'folder_result': '../results'
}

# using yfinance package: (we will likely not use)
#start = '2010-01-01'
#end   = '2011-01-01'
#stock = 'APPL'
#get_stocks(start, end, stock)

#-------------------------------------------------#
#               B) Data/Model/Loss                #
#-------------------------------------------------#
# B.1) dataset
#--------------------
# using manually downloaded data:
myStocks = pd.read_csv(hyperP['data_to_read'])
print(myStocks['Adj Close'])
myData = myStocks[['Close','Adj Close']]
# make data reasonable size
myData = myData/2000

# B.2) the model
#---------------
in_out_size = 2 #len(ALL_LETTERS)
myRnn = eval(hyperP['model'])(in_out_size,hyperP['hidden_size'],in_out_size)
# print the number of parameters
nbr_param = sum(p.numel() for p in myRnn.parameters() if p.requires_grad)
print("\n Model: "+hyperP['model']+" with ",nbr_param," parameters\n")
# use GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myRnn.to(device)
# B.3) loss (MSE) and optimizer (Adam)
#------------------------

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(myRnn.parameters(), lr=hyperP['lr'])

#-------------------------------------------------#
#                   C) Training                   #
#-------------------------------------------------#
t0 = time.time()
df_training = pd.DataFrame(columns=('step', 'loss'))
df_perplex = pd.DataFrame(columns=('step', 'perplexity'))
for step in range(hyperP['n_steps']):
    # C.1) initialize
    optimizer.zero_grad()
    # C.2) pick a sequence from the text...
    start_index = random.randint(0, len(myData) - hyperP['sequence_len'] - 2)
    end_index = start_index + hyperP['sequence_len'] + 1
    x_seq = myData[start_index:end_index].values
    x_seq_next = myData[(start_index+1):(end_index+1)].values
    # ...embed the sequence into a tensor
    x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32)#sequenceToTensor(x_seq,ALL_LETTERS)
    target = torch.tensor(x_seq_next, dtype=torch.float32)#torch.tensor( sequenceToIndex(x_seq_next,ALL_LETTERS) )
    # C.3) prediction
    y_seq_tensor,_ = myRnn(x_seq_tensor.unsqueeze(0).to(device)) # unsqueeze '0' for the size of the mini-batch
    # C.4) gradient step
    loss = criterion(y_seq_tensor.squeeze(0), target.to(device))
    loss.backward()
    optimizer.step()
    # C.5) save loss 
    df_training.loc[step] = [step, loss.item()]
    if (step%hyperP['print_every'] == 0):
        # print only once every 50 steps
        print('loss at step ',str(step),' : ', str(loss.item()))
        print("   input  = ", x_seq)
        #y_seq = x_seq[0]
        #for p in range(hyperP['sequence_len']-1):
        #    y_seq += y_seq_tensor[0,p,:]#tensorToLetter(y_seq_tensor[0,p,:],ALL_LETTERS)
        print("   output = ", y_seq_tensor)
        print(y_seq_tensor.size())
        y_seq = y_seq_tensor.detach().squeeze(0).numpy()
        plt.figure(0)
        plt.plot(y_seq[:,1]*2000)
        plt.plot(x_seq_next[:,1]*2000)
        plt.title('AMZN Stock Price Prediction vs Actual')
        plt.show()

    #if (step%hyperP['compute_perplexity_every'] == 0) & (step>0):
    #    perplexity = compute_perplexity(myText,ALL_LETTERS,myRnn)
    #    df_perplex.loc[len(df_perplex)] = [step, perplexity]
#-------------------------------------------------#
#              D) Plot/save results               #
#-------------------------------------------------#
time_elapsed = time.time() - t0
print(' Total time (s) : '+str(time_elapsed))
# create folder to plot/save result
str_time = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
nameFolder = hyperP['folder_result']+'/Report_'+str_time
os.makedirs(nameFolder)
# D.1) evolution loss/pexplexity
plt.figure(1);plt.clf()
plt.plot(df_training['step'],df_training['loss'],'-o',alpha=.2,label='loss')
#lowess = sm.nonparametric.lowess
#w = lowess(df_training['loss'], df_training['step'], frac=1/5)
#plt.plot(w[:,0],w[:,1],'-',color='green',linewidth=2,label='average loss')
plt.plot(df_perplex['step'],df_perplex['perplexity'],'-o',label='perplexity')
plt.grid()
plt.xlabel('steps')
plt.legend()
plt.title('loss/perplexity for the model '+hyperP['model'].replace('_',' '))
plt.savefig(nameFolder+'/evolution_loss_perplexity.pdf', bbox_inches='tight', pad_inches=0)
# D.2) saving data
# dataframes
df_training.to_csv(nameFolder+'/evolution_loss.csv',index=False)
df_perplex.to_csv(nameFolder+'/evolution_perplexity.csv',index=False)
# hyperparameters
hyperP['time_training'] = '{:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60)
with open(nameFolder+'/hyperParameters.json','w') as jsonFile:
    json.dump(hyperP, jsonFile, indent=2)
# the network
torch.save(myRnn.state_dict(),nameFolder+"/myModel.pth")

