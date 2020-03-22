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

# Train a RNN to output 'olleh' from 'hello':

class Vanilla_RNN(nn.Module):
    """ The vanilla RNN: from (x_t,h_t-1) input,hidden-state
    h_t = tanh( R*h_t-1 + A*x_t)
    y_t = B*h_t where A is the encoder, B the decoder, R the recurrent matrix """ 
    def __init__(self, input_size, hidden_size, output_size):
        super(Vanilla_RNN, self).__init__() 
        self.hidden_size = hidden_size 
        self.A = nn.Linear(input_size, hidden_size) # input to hidden 
        self.R = nn.Linear(hidden_size, hidden_size) # hidden to hidden 
        self.B = nn.Linear(hidden_size, output_size) # hidden to output
        self.tanh = nn.Tanh()
    def forward(self, x, h):
        # update the hidden state 
        h_update = self.tanh( self.R(h) + self.A(x) ) 
        # prediction 
        y = self.B(h_update) 
        return y,h_update
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


myRNN = Vanilla_RNN(4,2,4)

myText = 'hello'
goal = 'olleh'

import random 
def train_RNN(myRnn,myText,goal,P,shouldPrintTraining=False):
    """ Train a recurrent neural network from a text ('myText'). The dictionary P should contain:
    . the learning rate 'lr'
    . the number of steps 'n_steps'
    . the size of the sentence trained on 'chunk_len' """ 
    # init 
    optimizer = torch.optim.Adam(myRnn.parameters(), lr=P['lr']) 
    criterion = nn.CrossEntropyLoss()
    df = pd.DataFrame(columns=('step', 'loss'))
    # train 
    for step in range(P['n_steps']):
        #print('STEP: ', step)
        # A) initialize 
        h = myRnn.init_hidden() 
        optimizer.zero_grad() 
        loss = 0.0 
        # B) pick a chunk from the text 
        start_index = 0 # random.randint(0, len(myText) - P['chunk_len']) 
        end_index = start_index + P['chunk_len']
        chunk = myText[start_index:end_index] 
        if (shouldPrintTraining) & (step%500 == 0):
            print(" input = ", chunk)
            #chunk_predicted = chunk[0]
            chunk_predicted = '' 
        # C) prediction 
        for p in range(P['chunk_len']):
            # init
            x = letterToTensor( chunk[p] )
            #x_next = letterToTensor( chunk[p+1] )
            letter_x_next = letterToIndex(goal[p])
            #print(chunk[p+1])
            #print(letter_x_next)
            target = torch.tensor([letter_x_next],dtype=torch.long)
            # prediction
            y, h = myRnn(x, h)
            # loss
            loss += criterion(y.view(1,-1), target)
            #loss += criterion(y.view(1,-1), goal[Mod(p,5)])
            if (shouldPrintTraining):
                chunk_predicted += ALL_LETTERS[y.argmax()] 
            # D) gradient step 
            #if(p == 0):
            loss.backward(retain_graph=True) 
            #else:
            #    loss.backward() 
            optimizer.step() 
            # E) save loss 
            ave_loss = loss.detach().numpy() / P['chunk_len'] 
            if (shouldPrintTraining) & (step%500 == 0):
                print(" output = ", chunk_predicted)
            df.loc[step] = [step, ave_loss] 
            if (step%500 == 0):
                # print only once every 50 steps 
                print('loss at step ',str(step),' : ', str(ave_loss)) 
                if(p%4 == 0):
                #    print(myRnn.R.weight)
                    A = myRnn.A.weight
                    B = myRnn.B.weight
                    R = myRnn.R.weight
    # result 
    return df,A,B,R

P = {'n_steps' : 3000, 'lr' : .05, 'chunk_len' : 5} 
df,A,B,R = train_RNN(myRNN, myText, goal, P, True)


# Matrices taken from a particular successful train of the RNN:
R_b = torch.tensor([[ 4.2882, -2.7343],[ 2.5199,  0.0594]])
A_b = torch.tensor([[-1.7253,  0.3251,  1.2809,  2.1546],[ 2.6712, -1.6830, -2.0241,  1.6626]])
B_b = torch.tensor([[ 9.8825,  8.9663],[11.7690, -5.8867],[-9.0875, -8.4648],[-7.4939,  8.5177]])


X = 'hello'
h = torch.FloatTensor([[0],[0]])

y = []
for i in range(0,5): # i = 0:4
    x = letterToTensor(X[i])
    hpt = torch.mm(R_b,h) + torch.mm(A_b,x.view(4,1))
    h = tanh(hpt)
    y_tensor = torch.mm(B_b,h)
    y.append(ALL_LETTERS[y_tensor.argmax()])

# predicted letters:
print('Predicted letters part (b): ')
print(y)

# Matrices that give 'olleh' from 'hello' and 'h = [0,0]'
print('Matrices R, A, B: \n',R_b,'\n',A_b,'\n',B_b)
