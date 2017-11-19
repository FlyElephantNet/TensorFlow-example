
# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from numpy import *
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
import keras
import os

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=True)

os.system('tensorboard --logdir ./logs &')

with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

parallel_model = multi_gpu_model(model, gpus=8)


# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_test.shape
mean(X_train)

try:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
except Exception as err:
     print("exception ", format(err))
               
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
if (mean(X_train)>1):
    X_train /= 255
    X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
batch_size = 128
nb_classes = 10
llrate= 3e-5
nb_epoch = 100
regul= 1e-6
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
do= 0.2
model = Sequential()
model.add(Dense(500,input_shape=(X_train.shape[1],)))
#model.add(Dense(784, 128))
model.add(Activation('tanh'))
model.add(Dropout(do))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(do))
model.add(Dense(10))
model.add(Activation('softmax'))

rms= keras.optimizers.rmsprop(lr=llrate, decay=regul)
adm= keras.optimizers.Adam(lr=llrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=regul)
#model.compile(loss='mean_squared_error', optimizer=adm)
parallel_model.compile(loss='categorical_crossentropy', optimizer=adm)
parallel_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test), callbacks=[tensorboard])
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score)

model.save('cifarus10.h5')
