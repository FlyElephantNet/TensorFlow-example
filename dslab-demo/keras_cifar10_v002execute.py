# -*- coding: utf-8 -*-
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

import numpy as np
from PIL import Image
filename= 'cat1.jpg'
img = Image.open( filename )
try:
    data = np.asarray( img, dtype='float32' )
    img1= img.resize((32, 32), resample= Image.BICUBIC)
    obj= np.asarray( img1, dtype='float32' )
except SystemError:
    data = np.asarray( img.getdata(), dtype='float32' )


# In[4]:

vec= obj.reshape(1,32*32*3)/255.0


# In[6]:

batch_size= 128

prdc2= mdl2.predict(vec, batch_size= batch_size)


# In[7]:

cls= argmax(prdc2)


# In[9]:

clnames= ['airplane' , 'automobile', 'bird' , 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck' ]
print ('I think this isa class #', cls, clnames[cls])
