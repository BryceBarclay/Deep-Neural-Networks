# Primary file
import tensorflow as tf
from tensorflow import keras

class CNN_model(keras.Model):

    def __init__(self):
        super(CNN_model,self).__init__(name='cnn')
        self.conv1  = keras.layers.Conv1D(8, 4, strides=2, padding="valid", activation="relu", input_shape=(20, 4))
        self.conv2  = keras.layers.Conv1D(16, 4, strides=1, padding="valid", activation="relu")
        self.conv3  = keras.layers.Conv1D(32, 3, strides=1, padding="valid", activation="relu")
        self.flat   = keras.layers.Flatten(input_shape=(4, 32))
        self.dense1 = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        z1 = self.conv1(inputs)
        z2 = self.conv2(z1)
        z3 = self.conv3(z2)
        z4 = self.flat(z3)
        return self.dense1(z4)


class SimpleMLP_model(keras.Model):

    def __init__(self, use_bn=False, use_dp=True, num_classes=1):
        super(SimpleMLP_model, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)


class LSTM_model(keras.Model):
    
    def __init__(self, units = 20):
        super(LSTM_model,self).__init__(name='lstm')
        self.units = units
        self.LSTM1 = keras.layers.LSTM(units, return_sequences=True)
        self.LSTM2 = keras.layers.LSTM(20)
        self.dense1 = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.LSTM1(inputs)
        x = self.LSTM2(x)
        return self.dense1(x)


class GRU_model(keras.Model):
    
    def __init__(self, units = 20):
        super(GRU_model,self).__init__(name='gru')
        self.units = units
        self.GRU1 = keras.layers.GRU(units)
        self.dense1 = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.GRU1(inputs)
        return self.dense1(x)
