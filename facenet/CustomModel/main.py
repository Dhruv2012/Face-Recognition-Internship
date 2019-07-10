import cv2
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import keras
import glob
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
from keras.layers import Input, Layer, merge, concatenate
K.set_image_data_format('channels_first')
from keras.utils import plot_model
import pandas as pd
import os.path
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras.utils.data_utils import Sequence


TRAIN = 'D:/Summer Intern 2019/FACENET/testing/train_alignfix'
TEST = 'D:/Summer Intern 2019/FACENET/testing/test_alignfix'
linux = False
train_model = True             
scratch = False

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
    lossy = tf.reduce_mean(tf.maximum(loss,0.0))
    return lossy







def mytripletgenerator(path, batch):
    def nextFile(filename,directory):
        fileList = os.listdir(directory)
        nextIndex = fileList.index(filename) + 1
        if nextIndex == 0 or nextIndex == len(fileList):
            return None
        return fileList[nextIndex]

    def load_image(image_path):
        img1 = cv2.imread(image_path, 1)
        img1 = cv2.resize(img1, (96, 96))
        img = img1[...,::-1]
        img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=3)
        img_array = np.array([img])
        return img_array

    while True:
        images_a = np.zeros((batch, 3, 96, 96), dtype=np.float16)
        images_p = np.zeros((batch, 3,  96, 96),dtype=np.float16)
        images_n = np.zeros((batch, 3, 96, 96), dtype=np.float16)
        j=0
        

        img_ptr =0
        for dir, subdir, files in os.walk(path):
            breakBool = False
            
            present_dir = np.random.choice(os.listdir(path))
            file1 = np.random.choice(os.listdir(os.path.join(path,present_dir)))
            try:
                while(True):    
                    next_file = np.random.choice(os.listdir(os.path.join(path,present_dir)))
                    if(file1 == next_file):
                        continue
                    else:
                        break    
                file_path = (os.path.join(path,present_dir) + "/" + file1)
                nextfile_path = (os.path.join(path,present_dir) + "/" + next_file)
            except:
                continue



                    # a_batch
            
       
            img_a = load_image(file_path)
            images_a[img_ptr] = img_a
            del img_a                        

            img_p = load_image(nextfile_path)
            images_p[img_ptr] = img_p
            del img_p
            
                
            while(True):    
                random_dir = np.random.choice(os.listdir(path))   
                if(random_dir == present_dir):
                    continue
                else:
                    break
              
            random_dirpath = os.path.join(path,random_dir)
                
                #n_batch
            random_files=np.random.choice(os.listdir(random_dirpath))
            random_picked = (random_dirpath + "/" + random_files)
            #print("negative is: " + str(random_files) + " anchor is :" + str(file1) + "positive is: " + str(next_file))
                #print('\n')
            img_n = load_image(random_picked)                    
                #print(img_ptr)
            images_n[img_ptr] = img_n
            img_ptr += 1
            del img_n    
            j=j+1
                
            if j>=batch | img_ptr>=(batch-1):
                breakBool =True
                break
            if(breakBool):
                break
        a_batch = images_a[:img_ptr]
        p_batch = images_p[:img_ptr]
        n_batch = images_n[:img_ptr]
        z1  = np.random.rand(17,)
        z2  = np.random.rand(17,)
        z3  = np.random.rand(17,) 
        yield [a_batch , p_batch, n_batch], [z1, z2, z3]








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

'''
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
'''


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
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return identity



def fix(FRmodel):
    for layer in FRmodel.layers:
        layer.trainable = True
        print("layer" + str(layer) + " " +  str(layer.trainable))
    for layer in FRmodel.layers[:-3]:
        layer.trainable = False
        print("layer" + str(layer) + " " +  str(layer.trainable))



def fix_full(FRmodel):
    for layer in FRmodel.layers:
        layer.trainable = True
        print("layer" + str(layer) + " " +  str(layer.trainable))




################################################################################

###############################################################################

#Loading and testing on pretrained model
if (train_model == False):
    
    if(scratch ==  True) :
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        print("loading pre-trained weights from Tess..................")
        load_weights_from_FaceNet(FRmodel)
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

    else:
        
        FRmodel = load_model("mytraining.h5")
        FRmodel.load_weights("mytraining.h5")
        FRmodel.compile(optimizer='adam', loss = triplet_loss, metrics=['accuracy'])

    FRmodel.summary()
    metadata_train, database = load_metadata(TRAIN,FRmodel)
    print(metadata_train.shape)
    num_images = metadata_train.shape[0]

    #identity = recognise("D:\Summer Intern 2019\FACENET/testing/test\P17EC001/P17EC001_0043.jpg", database, FRmodel)
    #identity = recognise("D:\Summer Intern 2019\FACENET/testing/train_alignedv1\P17EC001/P17EC001_0004.jpg", database, FRmodel)

############################################################################################################################
##  CONFUSION MATRIX

    y_pred = []
    y_actual = []
    path = TEST

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
    #FRmodel.load_weights("mytraining.h5")
    FRmodel.summary()
    fix(FRmodel)
    FRmodel.summary()
    callbacks = [ModelCheckpoint('testing.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=20)]


    in_a = Input(shape=(3, 96, 96))
    in_p = Input(shape=(3, 96, 96))
    in_n = Input(shape=(3, 96, 96))
    emb_a = FRmodel(in_a)
    emb_p = FRmodel(in_p)
    emb_n = FRmodel(in_n)
    embeddings = [emb_a, emb_p, emb_n]
    class TripletLossLayer(Layer):
        def __init__(self, alpha, **kwargs):
            self.alpha = alpha
            super(TripletLossLayer, self).__init__(**kwargs)
        def triplet_loss(self, inputs):
            a, p, n = inputs
            p_dist = K.sum(K.square(a - p), axis=-1)
            n_dist = K.sum(K.square(a - n), axis=-1)
            return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss
    



    def triplet_loss_v3(y_true,y_pred):
        from keras import backend as K
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        #print("anchor")
        #print(anchor)
        #print(anchor.shape)
        
        
        post_dist = K.sum(K.square(anchor - positive), axis =-1)
        neg_dist = K.sum(K.square(anchor - negative), axis =-1)


        #print("post_dist")
        #print(post_dist)
        #print(post_dist.shape)
        
        loss = tf.add(0.3,tf.subtract(post_dist,neg_dist))
        
        tf.reshape(loss,[1,1])

        #total_loss = tf.reduce_mean(tf.maximum(loss,0.0),axis = 0)
        #print(total_loss)
        #print(total_loss.shape)
        
        #print("loss")
        #print(loss)
        #print(loss.shape)
        return loss 

    #triplet_loss_layer = TripletLossLayer(alpha=0.3, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    #triplet_loss_layer = merge(y_pred,mode=triplet_loss_v3, output_shape = (1,))
    embeddings = Concatenate(axis = -1)([emb_a, emb_p, emb_n])
    FRmodel_train = Model(input = [in_a, in_p, in_n], output = [emb_a, emb_p, emb_n]) 

    #FRmodel_train.get_layer('FaceRecoModel').load_weights('mytraining.h5')

    #a = np.random.rand(1,3,96,96)
    #b = np.random.rand(1,3,96,96)
    #c = np.random.rand(1,3,96,96)
    #z = FRmodel_train.predict([a,b,c])
    #print(z)
    #print(z.shape)    

    FRmodel_train.summary()
    train_generator = mytripletgenerator(TRAIN,32)
    test_generator = mytripletgenerator(TEST,32)
    Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #FRmodel_train.compile(loss= triplet_loss_v3, optimizer='adam')
    

    #ans = triplet_loss_v3(None,z)
    #print("functional call ans is: " + str(ans))
    FRmodel_train.compile(loss= triplet_loss_v3, optimizer='adam')
    
    FRmodel_train.fit_generator(train_generator, epochs= 20 ,steps_per_epoch=50, validation_data = test_generator, validation_steps = 50)
    

    #To save weights close to testing.h5 as they are the best weights(checkpoint) and also as the weights of testing.h5 cannot be copied directly to mytraining.h5 as both models are different i.e FRmodel_train and FRmodel
    #FRmodel_train.load_weights('testing.h5')
    #FRmodel_train.fit_generator(train_generator, epochs= 20 ,steps_per_epoch=50, validation_data = test_generator, validation_steps = 50)
    FRmodel_train.get_layer('FaceRecoModel').save('mytraining.h5')
    






    '''
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(FRmodel)
    #FRmodel.load_weights("mytraining.h5")
    FRmodel.summary()
    fix(FRmodel)
    FRmodel.summary()
    callbacks = [ModelCheckpoint('testing.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=20)]
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
            return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss
    
    triplet_loss_layer = TripletLossLayer(alpha=0.3, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    FRmodel_train = Model([in_a, in_p, in_n], triplet_loss_layer)
    
    #FRmodel_train.get_layer('FaceRecoModel').load_weights('mytraining.h5')
    
    FRmodel_train.summary()
    train_generator = mytripletgenerator(TRAIN,32)
    test_generator = mytripletgenerator(TEST,32)
    Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    FRmodel_train.compile(loss= None, optimizer='adam')
    
    
    FRmodel_train.fit_generator(train_generator, epochs= 2000 ,steps_per_epoch=50, validation_data = test_generator, validation_steps = 50, callbacks=callbacks)
    
    #To save weights close to testing.h5 as they are the best weights(checkpoint) and also as the weights of testing.h5 cannot be copied directly to mytraining.h5 as both models are different i.e FRmodel_train and FRmodel
    FRmodel_train.load_weights('testing.h5')
    FRmodel_train.fit_generator(train_generator, epochs= 2000 ,steps_per_epoch=50, validation_data = test_generator, validation_steps = 50, callbacks=callbacks)
    FRmodel_train.get_layer('FaceRecoModel').save('mytraining.h5')
'''








'''
    
    
    
### Keras version was 2.1.6
    
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(FRmodel)
    #FRmodel.load_weights("mytraining.h5")
    FRmodel.summary()
    fix(FRmodel)
    FRmodel.summary()
    #callbacks = [ModelCheckpoint('testing.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=20)]
    in_a = Input(shape=(3, 96, 96))
    in_p = Input(shape=(3, 96, 96))
    in_n = Input(shape=(3, 96, 96))
    emb_a = FRmodel(in_a)
    emb_p = FRmodel(in_p)
    emb_n = FRmodel(in_n)
    positive_dist = Lambda(euclidean_distance, name='pos_dist')([emb_a, emb_p])
    negative_dist = Lambda(euclidean_distance, name='neg_dist')([emb_a, emb_n])
    stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist])
    FRmodel_train = Model([in_a, in_p, in_n], stacked_dists, name='triple_siamese')
    
    FRmodel_train.summary()
    train_generator = mytripletgenerator(TRAIN,32)
    test_generator = mytripletgenerator(TEST,32)
    FRmodel_train.compile(optimizer = 'adam', loss = triplet_loss_v2)
    FRmodel_train.fit_generator(train_generator, epochs= 30 , steps_per_epoch=2, validation_data = test_generator, validation_steps = 2)
    FRmodel_train.get_layer('FaceRecoModel').save('mytraining.h5')
    
'''





'''
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    load_weights_from_FaceNet(FRmodel)
    #FRmodel.load_weights("mytraining.h5")
    FRmodel.summary()
    fix(FRmodel)
    FRmodel.summary()
    callbacks = [ModelCheckpoint('testing.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=20)]
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
            return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss
 
    def triplet_loss_v3(y_pred,alpha =0.3):
    #(None,128) encodings for anchor, positive, negative
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        post_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)  #sum across last axis
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        loss = tf.add(alpha,tf.subtract(post_dist,neg_dist))
        total_loss = tf.reduce_mean(tf.maximum(loss,0.0))
        return total_loss   
    
    def loss_v3(y_true, y_pred):
        return K.mean(y_pred) 
    
    #triplet_loss_layer = TripletLossLayer(alpha=0.3, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    triplet_loss_layer = merge(y_pred,mode=triplet_loss_v3, output_shape = (1,))
    FRmodel_train = Model(input = [in_a, in_p, in_n], output = triplet_loss_layer)
    
    
    
    FRmodel_train.summary()
    train_generator = mytripletgenerator(TRAIN,16)
    test_generator = mytripletgenerator(TEST,16)
    Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    FRmodel_train.compile(loss= loss_v3, optimizer='adam')
    
    FRmodel_train.fit_generator(train_generator, epochs= 20, steps_per_epoch=50, use_multiprocessing=True) 
    #To save weights close to testing.h5 as they are the best weights(checkpoint) and also as the weights of testing.h5 cannot be copied directly to mytraining.h5 as both models are different i.e FRmodel_train and FRmodel
    #FRmodel_train.load_weights('testing.h5')
    #FRmodel_train.fit_generator(train_generator, epochs= 20 , steps_per_epoch=50, use_multiprocessing=True)
    FRmodel_train.get_layer('FaceRecoModel').save('mytraining.h5')
    
'''