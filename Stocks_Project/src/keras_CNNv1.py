import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

path = (
    "/Users/Bryce/Desktop/Deep_net/Deep-Neural-Networks/Stocks_Project/dataset/"
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
