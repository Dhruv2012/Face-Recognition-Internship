<<<<<<< HEAD
import cv2
import numpy as np
#import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
import os
from keras import backend as K
#import time
import tensorflow as tf
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
from numpy import genfromtxt
from multiprocessing.dummy import Pool

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')
from keras.utils import plot_model
from numpy import genfromtxt
import pandas as pd


train_model = False   # For loading pre trained model or train it from scratch

def triplet_loss(y_true,y_pred,alpha =0.3):
    #(None,128) encodings for anchor, positive, negative
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    post_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)  #sum across last axis
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    loss = tf.add(alpha,tf.subtract(post_dist,neg_dist))
    total_loss = tf.reduce_sum(tf.maximum(loss,0.0))
    return total_loss



def load_database():
    database = {}
    for file in glob.glob("images/*.jpg"):
        (dirname, filename) = os.path.split(file)
        identity = os.path.splitext(filename)[0]
        database[identity] = img_to_encoding(file,FRmodel)
    for file in glob.glob("images/*.png"):
        (dirname1, filename1) = os.path.split(file)
        identity1 = os.path.splitext(filename1)[0]
        database[identity1] = img_to_encoding(file,FRmodel)
    return database
#MANUAL DATABASE

'''
def main():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    if train_model == False:
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(FRmodel)
    #else:
        # Train the model
    FRmodel.summary()
    plot_model(FRmodel, to_file='FRmodel.png')


main()
'''

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)
FRmodel.summary()
plot_model(FRmodel, to_file='FRmodel.png')
database = load_database()

'''
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
'''
###################################################################################



def verify(image_path, identity, database, model):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)

    encoding = img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(img_to_encoding(image_path, model) - database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", Face verified")
        door_open = True
    else:
        print("It's not " + str(identity) + ", Face not recognised")
        door_open = False
    return dist, door_open


def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


verify("images/test_image/camera_0.jpg", "younes", database, FRmodel)
who_is_it("images/test_image/camera_0.jpg", database, FRmodel)
=======
import cv2
import numpy as np
#import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
import os
from keras import backend as K
#import time
import tensorflow as tf
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
from numpy import genfromtxt
from multiprocessing.dummy import Pool

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')
from keras.utils import plot_model
from numpy import genfromtxt
import pandas as pd


train_model = False   # For loading pre trained model or train it from scratch

def triplet_loss(y_true,y_pred,alpha =0.3):
    #(None,128) encodings for anchor, positive, negative
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    post_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)  #sum across last axis
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    loss = tf.add(alpha,tf.subtract(post_dist,neg_dist))
    total_loss = tf.reduce_sum(tf.maximum(loss,0.0))
    return total_loss



def load_database():
    database = {}
    for file in glob.glob("images/*.jpg"):
        (dirname, filename) = os.path.split(file)
        identity = os.path.splitext(filename)[0]
        database[identity] = img_to_encoding(file,FRmodel)
    for file in glob.glob("images/*.png"):
        (dirname1, filename1) = os.path.split(file)
        identity1 = os.path.splitext(filename1)[0]
        database[identity1] = img_to_encoding(file,FRmodel)
    return database
#MANUAL DATABASE

'''
def main():
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    if train_model == False:
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(FRmodel)
    #else:
        # Train the model
    FRmodel.summary()
    plot_model(FRmodel, to_file='FRmodel.png')


main()
'''

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)
FRmodel.summary()
plot_model(FRmodel, to_file='FRmodel.png')
database = load_database()

'''
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
'''
###################################################################################



def verify(image_path, identity, database, model):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)

    encoding = img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(img_to_encoding(image_path, model) - database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", Face verified")
        door_open = True
    else:
        print("It's not " + str(identity) + ", Face not recognised")
        door_open = False
    return dist, door_open


def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


verify("images/test_image/camera_0.jpg", "younes", database, FRmodel)
who_is_it("images/test_image/camera_0.jpg", database, FRmodel)
>>>>>>> weights removed
