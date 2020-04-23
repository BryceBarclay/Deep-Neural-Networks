import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

path = (
    "/Users/socket778/Coding/APM598project/Deep-Neural-Networks/Stocks_Project/dataset/"
)
filename = "data.hdf5"

# get data
g = h5py.File(path + filename, "r")
print(g.keys())
train_input = g["inputs"][:]
train_labels = g["labels"][:]
test_input = g["inputs_test"][:]
test_labels = g["labels_test"][:]

# setup model
model = keras.Sequential(
    [keras.layers.LSTM(16), keras.layers.Dense(1, activation=tf.nn.sigmoid),]
)

model.compile(
    optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"]
)

print(train_input.shape)
print(train_labels.shape)
# train model
print(np.sum(train_labels) / train_labels.shape[0])
print(np.sum(test_labels) / test_labels.shape[0])
model.fit(train_input, train_labels, epochs=50, batch_size=32)
print(np.sum(train_labels) / train_labels.shape[0])

# evaluate
test_loss, test_acc = model.evaluate(test_input, test_labels)
print("test accuracy: ", test_acc)
print(np.sum(test_labels) / test_labels.shape[0])

# pred = model.predict_classes(test_input, verbose=1)
# print(pred)
