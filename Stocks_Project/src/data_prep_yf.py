# Primary file
# Set up data files and problem parameters

# this script reformats the data obtained from finam.ru (changes csv from a single entry to a table-like)
import csv
import numpy as np
import pandas as pd
import h5py
# personal libraries
from utils.toolbox import *
from utils.all_models import *

#-------------------------------------------------#
#               A) Problem parameters             #
#-------------------------------------------------#
probP = {
    'nbars_pred' : 20,
    'nbars_crit' : 5,
    'points'     : 0.7,
}

#-------------------------------------------------#
#               B) Files                          #
#-------------------------------------------------#
path = (
    "../dataset/"
)
files = ["AMZN.csv",
         "AAPL.csv",
         "FANG.csv",
         "MSFT.csv",
         "DIS.csv",
         "CI.csv",
         "TM.csv",
         "NFLX.csv",]
# files = ["train_short.csv", "test_short.csv"]
filename = "data.hdf5"
f = h5py.File(path + filename, "w")

#-------------------------------------------------#
#               c) labelling data                 #
#-------------------------------------------------#
test_flag = 1
cnt=0

inputs = np.zeros((1, probP['nbars_pred'], 4))
labels = np.zeros((1, 1))  

for filein in files:
    print("new file")
    prices = np.empty((1, 1, 4))
    with open(path + filein) as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=";")
        csv_reader = pd.read_csv(csv_file, delimiter=",")
    
        for row in csv_reader.iterrows(): #range(len(csv_reader)): #csv_reader:
            cnt = cnt + 1 
            row = row[1]
            line = np.array(
                (
                    [
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    # int(row[6]),
                    ]
                ),
                ndmin=3,
                )
            prices = np.append(prices, line, axis=1)

    prices = np.delete(prices, 0, 1)

    # print(int(np.floor(prices.shape[1] - probP['nbars_pred'] - probP['nbars_crit'])))
    for i in range(0, int(np.floor(prices.shape[1] - probP['nbars_pred'] - probP['nbars_crit']))):

        # first get the labels and
        max_list = [prices[0][i + probP['nbars_pred'] + 1][1]]
        for j in range(2, probP['nbars_crit']):
            max_list.append(prices[0][i + probP['nbars_pred'] + j][1])
        label = np.array(
            get_label(prices[0][i + probP['nbars_pred']][3], max_list, probP['points']), ndmin=2
        )
        labels = np.append(labels, label)

        # then normalize inputs
        # price_matrix = normalize(prices[:, i : i + probP['nbars_pred'], :])

        inputs = np.append(
            inputs,
            prices[:, i : i + probP['nbars_pred'], :] - prices[0][i + probP['nbars_pred'] - 1][3],
            axis=0,
        )
        #if i == 100:
        #    print(inputs[i][:][:])
    # delete the first (empty) row of both arrays which they were intialized with
    inputs = np.delete(inputs, 0, 0)
    labels = np.delete(labels, 0, 0)
    ## saving df to an h5 file

    # print(inputs.shape)
    # print(labels.shape)
    #if test_flag == 0:
        #print('train')
        #dset = f.create_dataset("inputs", data=inputs)
        #dset = f.create_dataset("labels", data=labels)
        #dset = f.require_dataset("inputs", data=inputs, shape=inputs.shape, dtype='float64')
        #dset = f.require_dataset("labels", data=labels, shape=labels.shape, dtype='float64')
    #else:

    if test_flag == 1: 
        print('test')
        #dset = f.create_dataset("inputs_test", data=inputs)
        #dset = f.create_dataset("labels_test", data=labels)
        dset = f.require_dataset("inputs_test", data=inputs, shape=inputs.shape, dtype='float64')
        dset = f.require_dataset("labels_test", data=labels, shape=labels.shape, dtype='float64')
        inputs = np.zeros((1, probP['nbars_pred'], 4))
        labels = np.zeros((1, 1)) 
    print(sum(labels) / labels.shape[0])
    print(inputs.shape)
    test_flag = 0

dset = f.require_dataset("inputs", data=inputs, shape=inputs.shape, dtype='float64')
dset = f.require_dataset("labels", data=labels, shape=labels.shape, dtype='float64')
g = h5py.File(path + filename, "r")
#print(g.keys())
# inputs = g["inputs"][:]
# labels = g["labels"][:]
print(len(g["inputs"][:]))
