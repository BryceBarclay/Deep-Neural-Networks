import numpy as np 
import pandas as pd 

# part (b) is still in progress


ALL_LETTERS = 'helo'
NB_LETTERS = len(ALL_LETTERS)

import torch 
def letterToIndex(letter):
    """ Find letter index from all_letters, e.g. "a" = 0 """ 
    return ALL_LETTERS.find(letter)

def letterToTensor(letter):
    """ Transform a letter into a 'hot-vector' (tensor) """ 
    #tensor = torch.zeros(1, NB_LETTERS,dtype=torch.long) 
    tensor = torch.zeros(1, NB_LETTERS) 
    tensor[0][letterToIndex(letter)] = 1 
    return tensor 

#print("Embedding of the character 'c':") # check if letterToTensor is implemented correctly
#print(letterToTensor('e'))


# (a)

# as defined in problem:
A = torch.FloatTensor([[1, -1, -0.5, 0.5], [1, 1, -0.5, -1]])
B = torch.FloatTensor([[1,1],[0.5,1],[-1,0],[0,-0.5]])
R = torch.FloatTensor([[1,0],[0,1]])
X = 'hello'
h = torch.FloatTensor([[0],[0]])

y = []
import torch.nn as nn 
tanh = nn.Tanh()
for i in range(0,5): # i = 0:4
    x = letterToTensor(X[i])
    hpt = torch.mm(R,h) + torch.mm(A,x.view(4,1))
    h = tanh(hpt)
    y_tensor = torch.mm(B,h)
    y.append(ALL_LETTERS[y_tensor.argmax()])

# predicted letters:
print('Predicted letters part (a): ')
print(y)


# (b)

A = torch.FloatTensor([[1, -1, -0.5, 0.5], [1, 1, -0.5, -1]])
B = torch.FloatTensor([[1,1],[0.5,1],[-1,0],[0,-0.5]])
R = torch.FloatTensor([[0,1],[1,0]])
X = 'hello'
h = torch.FloatTensor([[0],[0]])

y = []
for i in range(0,5): # i = 0:4
    x = letterToTensor(X[i])
    hpt = torch.mm(R,h) + torch.mm(A,x.view(4,1))
    h = tanh(hpt)
    y_tensor = torch.mm(B,h)
    y.append(ALL_LETTERS[y_tensor.argmax()])

# predicted letters:
print('Predicted letters part (b): ')
print(y)






