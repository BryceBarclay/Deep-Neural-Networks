# Organizing Keras version of stocks project

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

# classic libraries
import numpy as np
#import statsmodels.api as sm    # to estimate an average with 'loess'
import matplotlib.pyplot as plt
from matplotlib import rc
import random, string
import os, time, datetime, json

#-------------------------------------------------#
#               A) Hyper-parameters               #
#-------------------------------------------------#
hyperP = {
    'data_to_read': '../dataset/data.hdf5',
    'model': 'keras.layers.LSTM',  # 'LSTM', 'GRU' 
    'hidden_size': 40,
    'n_steps' : 2001,
    'lr' : .005,
    'sequence_len' : 100,   # x_1 x_2 ... x_100
    'epochs' : 4,
    #'print_every' : 200,
    #'compute_perplexity_every' : 1000,
    'folder_result': '../results'
}

#-------------------------------------------------#
#               B) Data/Model/loss                #
#-------------------------------------------------#
# get data
g = h5py.File(hyperP['data_to_read'],"r")#(path + filename, "r")
print(g.keys())
train_input = g["inputs"][:]
train_labels = g["labels"][:]
test_input = g["inputs_test"][:]
test_labels = g["labels_test"][:]
# setup model
model = keras.Sequential(
    [eval(hyperP['model'])(16), keras.layers.Dense(1, activation=tf.nn.sigmoid),]
)
model.compile(
    optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"]
)

#-------------------------------------------------#
#                   C) Training                   #
#-------------------------------------------------#
t0 = time.time()
csv_logger = keras.callbacks.CSVLogger(hyperP['folder_result']+'/training.log')

train_history = model.fit(train_input, train_labels, epochs=hyperP['epochs'], batch_size=32,callbacks=[csv_logger])

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
loss = train_history.history['loss']
plt.figure(1);plt.clf()
plt.plot(loss,'-o',alpha=.2,label='loss')
plt.grid()
plt.xlabel('steps')
plt.legend()
plt.title('loss/perplexity for the model '+hyperP['model'].replace('_',' '))
plt.show()
plt.savefig(nameFolder+'/evolution_loss_perplexity.pdf', bbox_inches='tight', pad_inches=0)



#plt.figure(1);plt.clf()
#plt.plot(df_training['step'],df_training['loss'],'-o',alpha=.2,label='loss')
#lowess = sm.nonparametric.lowess
#w = lowess(df_training['loss'], df_training['step'], frac=1/5)
#plt.plot(w[:,0],w[:,1],'-',color='green',linewidth=2,label='average loss')
#plt.plot(df_perplex['step'],df_perplex['perplexity'],'-o',label='perplexity')
#plt.grid()
#plt.xlabel('steps')
#plt.legend()
#plt.title('loss/perplexity for the model '+hyperP['model'].replace('_',' '))
#plt.savefig(nameFolder+'/evolution_loss_perplexity.pdf', bbox_inches='tight', pad_inches=0)




print(train_input.shape)
print(train_labels.shape)
print(np.sum(train_labels) / train_labels.shape[0])
print(np.sum(test_labels) / test_labels.shape[0])

print(np.sum(train_labels) / train_labels.shape[0])

# evaluate
test_loss, test_acc = model.evaluate(test_input, test_labels)
print("test accuracy: ", test_acc)
print(np.sum(test_labels) / test_labels.shape[0])




