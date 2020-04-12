# Pytorch libraries
import torch
import torch.nn as nn


class LSTM_RNN(nn.Module):
    """
    Long-Short-Term-Memory network.
    """    
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_RNN, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True) # input to hidden
        self.B = nn.Linear(hidden_size, output_size) # hidden to output
    
    def forward(self, x, *args):
        # update the hidden state
        h,(h_last,c_last) = self.LSTM(x, *args)
        # prediction
        y = self.B(h)

        return y,(h_last,c_last)
    

class GRU_RNN(nn.Module):
    """
    GRU network
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(input_size, hidden_size,  batch_first=True) # input to hidden
        self.B = nn.Linear(hidden_size, output_size) # hidden to output
    
    def forward(self, x, *args):
        h,h_last = self.GRU(x, *args)
        # prediction
        y = self.B(h)
        
        return y,h_last


