import cv2
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
#import dlib
import keras
import glob
#from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
import os
from keras import backend as K
import tensorflow as tf
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
from numpy import genfromtxt
from multiprocessing.dummy import Pool
from face import *
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.layers import Input, Layer
K.set_image_data_format('channels_first')
from keras.utils import plot_model
import pandas as pd
import os.path

train_model = False  # For loading pre trained model or train it from scratch
scratch = False


def triplet_loss(y_true,y_pred,alpha =0.3):
    #(None,128) encodings for anchor, positive, negative
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]


    post_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)  #sum across last axis
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    loss = tf.add(alpha,tf.subtract(post_dist,neg_dist))
    total_loss = tf.reduce_sum(tf.maximum(loss,0.0))
    return total_loss

def triplet_generator():

    while True:
        a_batch = np.random.rand(4, 3, 96, 96)
        p_batch = np.random.rand(4, 3,  96, 96)
        n_batch = np.random.rand(4, 3, 96, 96)
        yield [a_batch , p_batch, n_batch], None


def load_database():
    database = {}
    for file in glob.glob("images(face cropped)/*.jpg"):
        (dirname, filename) = os.path.split(file)
        identity = os.path.splitext(filename)[0]
        database[identity] = img_to_encoding(file,FRmodel)
    for file in glob.glob("images(face cropped)/*.png"):
        (dirname1, filename1) = os.path.split(file)
        identity1 = os.path.splitext(filename1)[0]
        database[identity1] = img_to_encoding(file,FRmodel)
    return database
#MANUAL DATABASE

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file
    def __repr__(self):
        return self.image_path()
    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path,model):
    metadata = []
    database = {}
    for i in os.listdir(path):
        print(i)
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
                database.setdefault(i,[]).append(img_to_encoding(str(IdentityMetadata(path, i, f)),model))
    return np.array(metadata), database



class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss



def verify(image_path, identity, database, model):

    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(img_to_encoding(image_path, model) - database[identity])
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

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - database[name])
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

def recognise(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    print("hey there")
    for name,value in database.items():
        print(name)
        for val in value:
            temp = [name,val]
            #print(database[i])
            #dist = np.linalg.norm(encoding - database[i])
            dist = np.linalg.norm(encoding - temp[1])
            if dist < min_dist:
                min_dist = dist
                identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return identity


#Loading and testing on pretrained model
if (train_model == False):
    if(scratch ==  True) :
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        load_weights_from_FaceNet(FRmodel)
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

    #FRmodel.load_weights('nn4.small2.v1.h5')
    #FRmodel.load_weights('mytraining.h5',by_name=True)

    else:
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        
        #FRmodel.load_weights('mytraining.h5')
        FRmodel.compile(optimizer='adam', loss = triplet_loss, metrics=['accuracy'])

    FRmodel.summary()
    plot_model(FRmodel, to_file='FRmodel.png')
    metadata_train, database = load_metadata('D:\Summer Intern 2019\FACENET/testing/train_alignedv1',FRmodel)
    print(metadata_train.shape)
    num_images = metadata_train.shape[0]


    identity = recognise("D:\Summer Intern 2019\FACENET/testing/test\P17EC001/P17EC001_0043.jpg", database, FRmodel)
    identity = recognise("D:\Summer Intern 2019\FACENET/testing/train_alignedv1\P17EC001/P17EC001_0004.jpg", database, FRmodel)

############################################################################################################################
##  CONFUSION MATRIX

    y_pred = []
    y_actual = []
    path = 'D:\Summer Intern 2019\FACENET/testing/test_alignedv1'

    for i in os.listdir(path):
        print(i)
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                y_pred.append(recognise(str(IdentityMetadata(path, i, f)), database, FRmodel))
                y_actual.append(i)
    print(y_pred)
    print(y_actual)
    cm = confusion_matrix(y_target=y_actual,
                          y_predicted=y_pred,
                          binary = False)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()

else:
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(FRmodel)
    #FRmodel.load_weights('mytraining.h5', by_name=True)
    FRmodel.summary()

    for layer in FRmodel.layers:
        layer.trainable = True
        print("layer" + str(layer) + " " +  str(layer.trainable))

    for layer in FRmodel.layers[:-3]:
        layer.trainable = False
        print("layer" + str(layer) + " " +  str(layer.trainable))
    FRmodel.summary()

    in_a = Input(shape=(3, 96, 96))
    in_p = Input(shape=(3, 96, 96))
    in_n = Input(shape=(3, 96, 96))
    emb_a = FRmodel(in_a)
    emb_p = FRmodel(in_p)
    emb_n = FRmodel(in_n)
    y_pred = [emb_a, emb_p, emb_n]

    class TripletLossLayer(Layer):
        def __init__(self, alpha, **kwargs):
            self.alpha = alpha
            super(TripletLossLayer, self).__init__(**kwargs)

        def triplet_loss(self, inputs):
            a, p, n = inputs
            p_dist = K.sum(K.square(a - p), axis=-1)
            n_dist = K.sum(K.square(a - n), axis=-1)
            return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss
    triple_loss = triplet_loss(None,y_pred)
    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    FRmodel_train = Model([in_a, in_p, in_n], triplet_loss_layer)
    FRmodel_train.summary()
    generator = triplet_generator()

    FRmodel_train.compile(loss= None, optimizer='adam')
    FRmodel_train.fit_generator(generator, epochs=100, steps_per_epoch=100)
    FRmodel_train.save_weights('mytraining.h5')
