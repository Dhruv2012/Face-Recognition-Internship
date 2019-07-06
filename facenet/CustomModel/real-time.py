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
from face import *

TRAIN = 'D:/Summer Intern 2019/FACENET/testing/train_alignfix'
TEST = 'D:/Summer Intern 2019/FACENET/testing/test_alignfix'
linux = False
if(linux):
    TRAIN = '/home/ml/FACENET/testing/train_alignfix'
    TEST  = '/home/ml/FACENET/testing/test_alignfix' 




def triplet_loss(y_true,y_pred,alpha =0.3):
    #(None,128) encodings for anchor, positive, negative
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    post_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)  #sum across last axis
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    loss = tf.add(alpha,tf.subtract(post_dist,neg_dist))
    total_loss = tf.reduce_sum(tf.maximum(loss,0.0))
    return total_loss

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

def img_to_encoding_realtime(img1,model):
    img1 = cv2.resize(img1, (96, 96))
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=3)
    x_train = np.array([img])
    embedding = model.predict(x_train)
    return embedding

def recognise_realtime(image, database, model):
    encoding = img_to_encoding_realtime(image, model)
    min_dist = 100
    #print("hey there")
    for name,value in database.items():
        #print(name)
        for val in value:
            temp = [name,val]
            #print(database[i])
            #dist = np.linalg.norm(encoding - database[i])
            dist = np.linalg.norm(encoding - temp[1])
            if dist < min_dist:
                min_dist = dist
                identity = name
    if min_dist > 0.6:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return identity

def realtime():
    #Loading model parameters
    FRmodel = load_model("mytraining.h5")
    FRmodel.load_weights("mytraining.h5")
    FRmodel.compile(optimizer='adam', loss = triplet_loss, metrics=['accuracy'])
    FRmodel.summary()

    #Loading the dataset
    metadata_train, database = load_metadata(TRAIN,FRmodel)
    print(metadata_train.shape)
    num_images = metadata_train.shape[0]

    #Setting Camera Configurations
    cap = cv2.VideoCapture(0)
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = cnn_detector(gray, 1)
        for i in faces:
            x = i.rect.left()
            y = i.rect.top()
            w = i.rect.right() - x
            h = i.rect.bottom() - y
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = align_mod(frame,i.rect)
            identity = recognise_realtime(face_img, database, FRmodel)
            cv2.putText(frame, str(identity), (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
        
        cv2.imshow('Video', frame)  
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"): # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()     

realtime()