import numpy as np 
import pandas as pd 
import torch 

# (b)

# as defined in problem: 
A = torch.FloatTensor([[1,0],[0,1]])
B = torch.FloatTensor([[1,0],[0,1]])
R = torch.FloatTensor([[0.5,-1],[-1,0.5]])
h = torch.FloatTensor([[0],[0]])

# x = [0,0]^T:
x = torch.FloatTensor([[0],[0]])
y = []
import torch.nn as nn 
tanh = nn.Tanh()
for i in range(0,30): # i = 0:29
    hpt = torch.mm(R,h) + torch.mm(A,x)
    h = tanh(hpt)
    y_tensor = torch.mm(B,h)

# predicted tensor:
print('Predicted y for part (b): ')
print(y_tensor)

# x = [eps,-eps]^T:
y_eps = []
y30_diff = []
eps_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
for i in range(0,len(eps_list)):
    x_eps = torch.FloatTensor([[eps_list[i]],[-eps_list[i]]])
    h = torch.FloatTensor([[0],[0]])
    hpt = torch.mm(R,h) + torch.mm(A,x_eps)
    h = tanh(hpt)
    for k in range(0,29): # k = 0:28
        hpt = torch.mm(R,h) + torch.mm(A,torch.FloatTensor([[0],[0]]))
        h = tanh(hpt)
    y_tensor_eps = torch.mm(B,h)
    y_eps.append(y_tensor_eps)
    diff = y_tensor - y_eps[i]
    #diff = diff.tolist()
    #y30_diff.append(np.abs(diff[0]) + np.abs(diff[1]))
    y30_diff.append(torch.norm(diff))


import matplotlib.pyplot as plt
plt.plot(np.log10(eps_list),np.log10(y30_diff))
plt.title('log-log difference (b)')
plt.ylabel('log(||y30 - y30eps||)')
plt.xlabel('log(eps)')
plt.show()


# (c)

# as defined in problem: 
A = torch.FloatTensor([[1,0],[0,1]])
B = torch.FloatTensor([[1,0],[0,1]])
R = torch.FloatTensor([[0.5,-1],[-1,0.5]])
h = torch.FloatTensor([[0],[0]])

# x = [2,1]^T:
x = torch.FloatTensor([[2],[1]])
y = []
tanh = nn.Tanh()
for i in range(0,30): # i = 0:29
    hpt = torch.mm(R,h) + torch.mm(A,x)
    h = tanh(hpt)
    y_tensor = torch.mm(B,h)
    x = torch.FloatTensor([[0],[0]])

# predicted tensor:
print('Predicted y for part (c): ')
print(y_tensor)

# x = [2+eps,1-eps]^T:
y_eps = []
y30_diff = []
eps_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
for i in range(0,len(eps_list)):
    x_eps = torch.FloatTensor([[2+eps_list[i]],[1-eps_list[i]]])
    h = torch.FloatTensor([[0],[0]])
    hpt = torch.mm(R,h) + torch.mm(A,x_eps)
    h = tanh(hpt)
    for k in range(0,29): # k = 0:28
        hpt = torch.mm(R,h) + torch.mm(A,torch.FloatTensor([[0],[0]]))
        h = tanh(hpt)
    y_tensor_eps = torch.mm(B,h)
    y_eps.append(y_tensor_eps)
    diff = y_tensor - y_eps[i]
    #diff = diff.tolist()
    #y30_diff.append(np.abs(diff[0]) + np.abs(diff[1]))
    y30_diff.append(torch.norm(diff))


plt.plot(np.log10(eps_list),np.log10(y30_diff))
plt.title('log-log difference (c)')
plt.ylabel('log(||y30 - y30eps||)')
plt.xlabel('log(eps)')
plt.show()


# (Extra)

# as defined in problem: 
A = torch.FloatTensor([[1,0],[0,1]])
B = torch.FloatTensor([[1,0],[0,1]])
R = torch.FloatTensor([[0.5,-1],[-1,0.5]])
h = torch.FloatTensor([[0],[0]])

# x = [0,0]^T:
x = torch.FloatTensor([[0],[0]])
y = []
tanh = nn.Tanh()
for i in range(0,30): # i = 0:29
    hpt = torch.mm(R,h) + torch.mm(A,x)
    h = tanh(hpt)
    y_tensor = torch.mm(B,h)

# predicted tensor:
print('Predicted y for part (Extra): ')
print(y_tensor)

# x = [eps,eps]^T:
y_eps = []
y30_diff = []
eps_list = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
for i in range(0,len(eps_list)):
    x_eps = torch.FloatTensor([[eps_list[i]],[eps_list[i]]])
    h = torch.FloatTensor([[0],[0]])
    hpt = torch.mm(R,h) + torch.mm(A,x_eps)
    h = tanh(hpt)
    for k in range(0,29): # k = 0:28
        hpt = torch.mm(R,h) + torch.mm(A,torch.FloatTensor([[0],[0]]))
        h = tanh(hpt)
    y_tensor_eps = torch.mm(B,h)
    y_eps.append(y_tensor_eps)
    diff = y_tensor - y_eps[i]
    #diff = diff.tolist()
    #y30_diff.append(np.abs(diff[0]) + np.abs(diff[1]))
    y30_diff.append(torch.norm(diff))


plt.plot(np.log10(eps_list),np.log10(y30_diff))
plt.title('log-log difference (Extra)')
plt.ylabel('log(||y30 - y30eps||)')
plt.xlabel('log(eps)')
plt.show()
