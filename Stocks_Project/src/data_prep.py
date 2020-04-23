# this script reformats the data obtained from finam.ru (changes csv from a single entry to a table-like)

import csv
import numpy as np
import pandas as pd
import h5py

path = (
    "/Users/socket778/Coding/APM598project/Deep-Neural-Networks/Stocks_Project/dataset/"
)
files = ["SBER_170605_190531.csv", "SBER_190603_190927.csv"]
# files = ["train_short.csv", "test_short.csv"]
filename = "data.hdf5"
f = h5py.File(path + filename, "w")

nbars_pred = 20
nbars_crit = 5
points = 0.7


# function assigning labels based on max values of the next nbars_crit candles
def get_label(close, max_list, pts):
    if close + pts < np.amax(max_list):
        label = 1
        # print(close, np.amax(max_list))
    else:
        label = 0
    return label


# function normalizing the input matrix of prices
# basically subtracts the starting price from each row


test_flag = 0
for filein in files:
    prices = np.empty((1, 1, 4))
    with open(path + filein) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        cnt = 0
        for row in csv_reader:
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
    # print(prices[0])

    # print(prices.shape)

    inputs = np.zeros((1, nbars_pred, 4))
    labels = np.zeros((1, 1))  # []
    # print(int(np.floor(prices.shape[1] - nbars_pred - nbars_crit)))
    for i in range(0, int(np.floor(prices.shape[1] - nbars_pred - nbars_crit))):

        # first get the labels and
        max_list = [prices[0][i + nbars_pred + 1][1]]
        for j in range(2, nbars_crit):
            max_list.append(prices[0][i + nbars_pred + j][1])
        label = np.array(
            get_label(prices[0][i + nbars_pred][3], max_list, points), ndmin=2
        )
        labels = np.append(labels, label)

        # then normalize inputs
        # price_matrix = normalize(prices[:, i : i + nbars_pred, :])

        inputs = np.append(
            inputs,
            prices[:, i : i + nbars_pred, :] - prices[0][i + nbars_pred - 1][3],
            axis=0,
        )
        if i == 100:
            print(inputs[i][:][:])
    # delete the first (empty) row of both arrays which they were intialized with
    inputs = np.delete(inputs, 0, 0)
    labels = np.delete(labels, 0, 0)
    ## saving df to an h5 file

    # print(inputs.shape)
    # print(labels.shape)
    if test_flag == 0:
        dset = f.create_dataset("inputs", data=inputs)
        dset = f.create_dataset("labels", data=labels)
    else:
        dset = f.create_dataset("inputs_test", data=inputs)
        dset = f.create_dataset("labels_test", data=labels)
    print(sum(labels) / labels.shape[0])
    test_flag = 1


g = h5py.File(path + filename, "r")
print(g.keys())
# inputs = g["inputs"][:]
# labels = g["labels"][:]
