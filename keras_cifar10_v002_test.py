# -*- coding: utf-8 -*-
# In[50]:

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
import keras

from keras.models import load_model
mdl2= load_model('cifarus10.h5')


# In[51]:

nclass=10
prdc= mdl2.predict(X_test, batch_size= batch_size)
score = mdl2.evaluate(X_test, Y_test,  verbose=1)

cmatr= zeros([nclass, nclass])
for ii in range(10000):
    relv= argmax(Y_test[ii,:])
    predv= argmax(prdc[ii, :])
    cmatr[relv, predv]= cmatr[relv, predv]+1;
print('CIFAR 10 confusion matrix')    
print (cmatr)



# In[52]:

errate= 100.0- sum(diag(cmatr))/100


# In[53]:

print('Error rate', errate)


