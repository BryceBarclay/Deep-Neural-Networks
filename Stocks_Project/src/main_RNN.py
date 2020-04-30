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
    'model': 'CNN', #'keras.layers.LSTM',  # 'LSTM', 'GRU', 'CNN'
    'units': 40,
    'lr' : .005,
    'epochs' : 1,
    'folder_result': '../results'
}

#-------------------------------------------------#
#               B) Data/Model/loss                #
#-------------------------------------------------#
# get data
g = h5py.File(hyperP['data_to_read'],"r")
train_input = g["inputs"][:]
train_labels = g["labels"][:]
test_input = g["inputs_test"][:]
test_labels = g["labels_test"][:]
# setup model
if(hyperP['model'] == 'CNN'):
    model = keras.Sequential(
    [
        keras.layers.Conv1D(
            8, 4, strides=2, padding="valid", activation="relu", input_shape=(20, 4)
        ),
        keras.layers.Conv1D(16, 4, strides=1, padding="valid", activation="relu"),
        keras.layers.Conv1D(32, 3, strides=1, padding="valid", activation="relu"),
        keras.layers.Flatten(input_shape=(4, 32)),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
    )
else:
    model = keras.Sequential(
        [eval(hyperP['model'])(hyperP['units']), keras.layers.Dense(1, activation=tf.nn.sigmoid),]
    )

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

with open(nameFolder+'/hyperParameters.json','w') as jsonFile:
    json.dump(hyperP, jsonFile, indent=2)
json_string = model.to_json()
with open(nameFolder+'/architecture.json','w') as jsonFile:
    json.dump(json_string, jsonFile, indent=2)

# evaluate on test
test_loss, test_acc = model.evaluate(test_input, test_labels)
print("test accuracy: ", test_acc)

with open(nameFolder+'/training.csv','a') as fd:
    fd.write('Test: ,' + str(test_acc))

print(model.summary())

