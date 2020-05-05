# Primary file
# Trining, testing, and saving results of keras models including CNN, LSTM, GRU

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

# personal libraries
from utils.toolbox import *
from utils.keras_models import *

#-------------------------------------------------#
#               A) Hyper-parameters               #
#-------------------------------------------------#
hyperP = {
    'data_to_read': '../dataset/data.hdf5',
    'model': 'LSTM',  # 'LSTM', 'GRU', 'CNN', 'SimpleMLP'
    'units': 40, # RNN only
    'lr' : .005,
    'epochs' : 5,
    'folder_result': '../results'
}
# use utils.keras_models classes:
hyperP['model'] = hyperP['model'] + '_model'

#-------------------------------------------------#
#               B) Data/Model/loss                #
#-------------------------------------------------#
# get data
g = h5py.File(hyperP['data_to_read'],"r")
train_input = g["inputs"][:]
train_labels = g["labels"][:]
test_input = g["inputs_test"][:]
test_labels = g["labels_test"][:]
print(test_labels.shape)
# setup model using utils.keras_models
RNN = False
if((hyperP['model'] == 'GRU') | (hyperP['model'] == 'LSTM')):
    RNN = True
if(RNN):
    model = eval(hyperP['model'])(hyperP['units'])
else:
    model = eval(hyperP['model'])()
# optimizer and compile
optim = keras.optimizers.Adam(lr = hyperP['lr'])
model.compile(
    optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"]
)

#-------------------------------------------------#
#                   C) Training                   #
#-------------------------------------------------#
# create folder to plot/save result
str_time = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
nameFolder = hyperP['folder_result']+'/Report_'+str_time
os.makedirs(nameFolder)
csv_logger = keras.callbacks.CSVLogger(nameFolder+'/training.csv')

t0 = time.time()
# Train
train_history = model.fit(train_input, train_labels, epochs=hyperP['epochs'], batch_size=32,callbacks=[csv_logger])

#-------------------------------------------------#
#              D) Plot/save results               #
#-------------------------------------------------#
time_elapsed = time.time() - t0
print(' Total time (s) : '+str(time_elapsed))
# D.1) evolution of loss
loss = train_history.history['loss']
plt.figure(1);plt.clf()
plt.plot(loss,'-o',alpha=.2,label='loss')
plt.grid()
plt.xlabel('steps')
plt.legend()
plt.title('loss for the model '+hyperP['model'].replace('_',' '))
plt.savefig(nameFolder+'/evolution_loss_perplexity.pdf', bbox_inches='tight', pad_inches=0)

# D.2) saving parameters 
with open(nameFolder+'/hyperParameters.json','w') as jsonFile:
    json.dump(hyperP, jsonFile, indent=2)

# evaluate on test
test_loss, test_acc = model.evaluate(test_input, test_labels)
print("test accuracy: ", test_acc)

with open(nameFolder+'/training.csv','a') as fd:
    fd.write('Test: ,' + str(test_acc))

print(model.summary())

