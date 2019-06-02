<<<<<<< HEAD
'''
IMPROVEMENTS THAT SHOULD BE DONE:
               1. Generalized dataset load code(CSV,from libraries)
               2. Standardizing the dataset and converting into 4D arrsys
               3. Data Augmentation
               4. Dropout
'''


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



'''
LOADING THE DATASET & NORMALIZING OR STANDARDIZING IT
'''

# Let's take here the Famous MNIST dataset
# This part will vary depending on the dataset file extension or either it is imported locally or from libraries

(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()

# Normalizing the vectors

x_train = x_train_orig/255
x_test = x_test_orig/255            # FOR IMAGE DATASET
y_train = y_train_orig
y_test = y_test_orig

# Check the shape of the X and Y . If X is not 4 Dimensional,Reshape it into 4 Dimensional(X) . For Y , it should be a row  vector


input_shape = (x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test =  x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# VISUALIZING THE DATASET
print ("number of training examples = " + str(x_train.shape[0]))
print ("number of test examples = " + str(x_test.shape[0]))
print ("X_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))
print("INPUT Shape is :" + str(input_shape))

def MODEL(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0', kernel_initializer='glorot_uniform', bias_initializer='zeros')(X_input)   #kernel_initializer = he_normal  for deeper networks
    X = BatchNormalization(axis=3, name='bn0')(X)                                                                                 #Refs:- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=None, name='max_pool0')(X)

    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1', kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=None, name='max_pool1')(X)


    #X = Conv2D(128 , (3, 3), strides=(2, 2), name='conv2')(X)
    #X = BatchNormalization(axis=3, name='bn2')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D((2, 2), strides=None, name='max_pool2')(X)

    X = Flatten()(X)
    X = Dense(100, activation='relu', name='fc3')(X)
    X = Dense(10, activation='softmax', name='fc4')(X)

    model = Model(inputs=X_input, outputs=X, name='genCNNModel')
    return  model

Model =  MODEL(input_shape)
Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])


sample_count = x_train.shape[0]
batch_size = 32
steps_per_epoch = sample_count // batch_size
use_batch_size = True

if use_batch_size:
    Model.fit(x_train, y_train, batch_size = batch_size, epochs = 2)
else:
    Model.fit(x_train, y_train, steps_per_epoch = steps_per_epoch, epochs= 2)
test_loss, test_acc = Model.evaluate(x_test, y_test)

print('TEST LOSS:' , test_loss)
print('TEST ACCURACY:' , test_acc)
Model.summary()
=======
'''
IMPROVEMENTS THAT SHOULD BE DONE:
               1. Generalized dataset load code(CSV,from libraries)
               2. Standardizing the dataset and converting into 4D arrsys
               3. Data Augmentation
               4. Dropout
'''


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



'''
LOADING THE DATASET & NORMALIZING OR STANDARDIZING IT
'''

# Let's take here the Famous MNIST dataset
# This part will vary depending on the dataset file extension or either it is imported locally or from libraries

(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.mnist.load_data()

# Normalizing the vectors

x_train = x_train_orig/255
x_test = x_test_orig/255            # FOR IMAGE DATASET
y_train = y_train_orig
y_test = y_test_orig

# Check the shape of the X and Y . If X is not 4 Dimensional,Reshape it into 4 Dimensional(X) . For Y , it should be a row  vector


input_shape = (x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test =  x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# VISUALIZING THE DATASET
print ("number of training examples = " + str(x_train.shape[0]))
print ("number of test examples = " + str(x_test.shape[0]))
print ("X_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))
print("INPUT Shape is :" + str(input_shape))

def MODEL(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0', kernel_initializer='glorot_uniform', bias_initializer='zeros')(X_input)   #kernel_initializer = he_normal  for deeper networks
    X = BatchNormalization(axis=3, name='bn0')(X)                                                                                 #Refs:- https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=None, name='max_pool0')(X)

    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1', kernel_initializer='glorot_uniform', bias_initializer='zeros')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=None, name='max_pool1')(X)


    #X = Conv2D(128 , (3, 3), strides=(2, 2), name='conv2')(X)
    #X = BatchNormalization(axis=3, name='bn2')(X)
    #X = Activation('relu')(X)
    #X = MaxPooling2D((2, 2), strides=None, name='max_pool2')(X)

    X = Flatten()(X)
    X = Dense(100, activation='relu', name='fc3')(X)
    X = Dense(10, activation='softmax', name='fc4')(X)

    model = Model(inputs=X_input, outputs=X, name='genCNNModel')
    return  model

Model =  MODEL(input_shape)
Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics = ['accuracy'])


sample_count = x_train.shape[0]
batch_size = 32
steps_per_epoch = sample_count // batch_size
use_batch_size = True

if use_batch_size:
    Model.fit(x_train, y_train, batch_size = batch_size, epochs = 2)
else:
    Model.fit(x_train, y_train, steps_per_epoch = steps_per_epoch, epochs= 2)
test_loss, test_acc = Model.evaluate(x_test, y_test)

print('TEST LOSS:' , test_loss)
print('TEST ACCURACY:' , test_acc)
Model.summary()
>>>>>>> weights removed
plot_model(Model, to_file='Model.png')