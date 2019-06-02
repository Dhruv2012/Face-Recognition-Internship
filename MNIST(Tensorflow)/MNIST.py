<<<<<<< HEAD
import tensorflow as tf
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt
#matplotlib inline # Only use this if using iPython
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
print(x_train.shape)


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(y_test[4444])
print(y_test[1])
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('TEST LOSS:' , test_loss)
print('TEST ACCURACY:' , test_acc)
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())



#Analysis with Convolution and pooling layers
'''
3(Pair of Convolution and pooling layers)
32,64,128

60000/60000 [==============================] - 19s 313us/step - loss: 0.2022 - acc: 0.9381
Epoch 2/20
60000/60000 [==============================] - 16s 264us/step - loss: 0.0745 - acc: 0.9768
Epoch 3/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0582 - acc: 0.9820
Epoch 4/20
60000/60000 [==============================] - 31s 515us/step - loss: 0.0475 - acc: 0.9852
Epoch 5/20
60000/60000 [==============================] - 44s 735us/step - loss: 0.0408 - acc: 0.9871
Epoch 6/20
60000/60000 [==============================] - 44s 736us/step - loss: 0.0341 - acc: 0.9893
Epoch 7/20
60000/60000 [==============================] - 44s 732us/step - loss: 0.0315 - acc: 0.9899
Epoch 8/20
60000/60000 [==============================] - 44s 733us/step - loss: 0.0274 - acc: 0.9912
Epoch 9/20
60000/60000 [==============================] - 45s 753us/step - loss: 0.0297 - acc: 0.9908
Epoch 10/20
60000/60000 [==============================] - 47s 790us/step - loss: 0.0221 - acc: 0.9928
Epoch 11/20
60000/60000 [==============================] - 39s 651us/step - loss: 0.0232 - acc: 0.9927
Epoch 12/20
60000/60000 [==============================] - 25s 412us/step - loss: 0.0226 - acc: 0.9929
Epoch 13/20
60000/60000 [==============================] - 24s 400us/step - loss: 0.0213 - acc: 0.9933
Epoch 14/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0214 - acc: 0.9939
Epoch 15/20
60000/60000 [==============================] - 18s 302us/step - loss: 0.0207 - acc: 0.9938
Epoch 16/20
60000/60000 [==============================] - 20s 330us/step - loss: 0.0190 - acc: 0.9942
Epoch 17/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0174 - acc: 0.9949
Epoch 18/20
60000/60000 [==============================] - 20s 331us/step - loss: 0.0201 - acc: 0.9948
Epoch 19/20
60000/60000 [==============================] - 20s 333us/step - loss: 0.0175 - acc: 0.9949
Epoch 20/20
60000/60000 [==============================] - 19s 321us/step - loss: 0.0177 - acc: 0.9952
10000/10000 [==============================] - 1s 119us/step
TEST LOSS: 0.08804376618057431
TEST ACCURACY: 0.986
9
     
2 ->   32,64


60000/60000 [==============================] - 18s 302us/step - loss: 0.1498 - acc: 0.9545
Epoch 2/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0540 - acc: 0.9832
Epoch 3/20
60000/60000 [==============================] - 44s 741us/step - loss: 0.0433 - acc: 0.9864
Epoch 4/20
60000/60000 [==============================] - 46s 765us/step - loss: 0.0335 - acc: 0.9893
Epoch 5/20
60000/60000 [==============================] - 33s 545us/step - loss: 0.0285 - acc: 0.9906
Epoch 6/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0223 - acc: 0.9929
Epoch 7/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0211 - acc: 0.9933
Epoch 8/20
60000/60000 [==============================] - 20s 339us/step - loss: 0.0203 - acc: 0.9932
Epoch 9/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0185 - acc: 0.9941
Epoch 10/20
60000/60000 [==============================] - 21s 355us/step - loss: 0.0162 - acc: 0.9947
Epoch 11/20
60000/60000 [==============================] - 21s 355us/step - loss: 0.0160 - acc: 0.9950
Epoch 12/20
60000/60000 [==============================] - 22s 367us/step - loss: 0.0128 - acc: 0.9960
Epoch 13/20
60000/60000 [==============================] - 20s 340us/step - loss: 0.0145 - acc: 0.9955
Epoch 14/20
60000/60000 [==============================] - 20s 329us/step - loss: 0.0132 - acc: 0.9957
Epoch 15/20
60000/60000 [==============================] - 19s 321us/step - loss: 0.0135 - acc: 0.9959
Epoch 16/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0114 - acc: 0.9965
Epoch 17/20
60000/60000 [==============================] - 23s 377us/step - loss: 0.0126 - acc: 0.9962
Epoch 18/20
60000/60000 [==============================] - 19s 318us/step - loss: 0.0121 - acc: 0.9963
Epoch 19/20
60000/60000 [==============================] - 20s 326us/step - loss: 0.0110 - acc: 0.9969
Epoch 20/20
60000/60000 [==============================] - 20s 337us/step - loss: 0.0142 - acc: 0.9959
10000/10000 [==============================] - 1s 124us/step
TEST LOSS: 0.053206579726541164
TEST ACCURACY: 0.9906
9

1-> 32

        60000/60000 [==============================] - 25s 416us/step - loss: 0.2055 - acc: 0.9380
Epoch 2/20
60000/60000 [==============================] - 21s 346us/step - loss: 0.0810 - acc: 0.9752
Epoch 3/20
60000/60000 [==============================] - 23s 378us/step - loss: 0.0578 - acc: 0.9816
Epoch 4/20
60000/60000 [==============================] - 21s 351us/step - loss: 0.0439 - acc: 0.9862
Epoch 5/20
60000/60000 [==============================] - 21s 356us/step - loss: 0.0338 - acc: 0.9892
Epoch 6/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0284 - acc: 0.9907
Epoch 7/20
60000/60000 [==============================] - 22s 360us/step - loss: 0.0241 - acc: 0.9918
Epoch 8/20
60000/60000 [==============================] - 21s 354us/step - loss: 0.0217 - acc: 0.9929
Epoch 9/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0202 - acc: 0.9934
Epoch 10/20
60000/60000 [==============================] - 20s 334us/step - loss: 0.0162 - acc: 0.9944
Epoch 11/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0179 - acc: 0.9941
Epoch 12/20
60000/60000 [==============================] - 21s 353us/step - loss: 0.0149 - acc: 0.9949
Epoch 13/20
60000/60000 [==============================] - 21s 344us/step - loss: 0.0165 - acc: 0.9943
Epoch 14/20
60000/60000 [==============================] - 21s 348us/step - loss: 0.0131 - acc: 0.9956
Epoch 15/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0135 - acc: 0.9955
Epoch 16/20
60000/60000 [==============================] - 21s 350us/step - loss: 0.0128 - acc: 0.9958
Epoch 17/20
60000/60000 [==============================] - 20s 341us/step - loss: 0.0135 - acc: 0.9957
Epoch 18/20
60000/60000 [==============================] - 21s 343us/step - loss: 0.0127 - acc: 0.9958
Epoch 19/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0122 - acc: 0.9960
Epoch 20/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0088 - acc: 0.9969
10000/10000 [==============================] - 1s 109us/step
TEST LOSS: 0.07443016861326614
TEST ACCURACY: 0.9855
'''

=======
import tensorflow as tf
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
import matplotlib.pyplot as plt
#matplotlib inline # Only use this if using iPython
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
print(x_train.shape)


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(y_test[4444])
print(y_test[1])
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, kernel_size=(3,3), input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('TEST LOSS:' , test_loss)
print('TEST ACCURACY:' , test_acc)
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())



#Analysis with Convolution and pooling layers
'''
3(Pair of Convolution and pooling layers)
32,64,128

60000/60000 [==============================] - 19s 313us/step - loss: 0.2022 - acc: 0.9381
Epoch 2/20
60000/60000 [==============================] - 16s 264us/step - loss: 0.0745 - acc: 0.9768
Epoch 3/20
60000/60000 [==============================] - 22s 368us/step - loss: 0.0582 - acc: 0.9820
Epoch 4/20
60000/60000 [==============================] - 31s 515us/step - loss: 0.0475 - acc: 0.9852
Epoch 5/20
60000/60000 [==============================] - 44s 735us/step - loss: 0.0408 - acc: 0.9871
Epoch 6/20
60000/60000 [==============================] - 44s 736us/step - loss: 0.0341 - acc: 0.9893
Epoch 7/20
60000/60000 [==============================] - 44s 732us/step - loss: 0.0315 - acc: 0.9899
Epoch 8/20
60000/60000 [==============================] - 44s 733us/step - loss: 0.0274 - acc: 0.9912
Epoch 9/20
60000/60000 [==============================] - 45s 753us/step - loss: 0.0297 - acc: 0.9908
Epoch 10/20
60000/60000 [==============================] - 47s 790us/step - loss: 0.0221 - acc: 0.9928
Epoch 11/20
60000/60000 [==============================] - 39s 651us/step - loss: 0.0232 - acc: 0.9927
Epoch 12/20
60000/60000 [==============================] - 25s 412us/step - loss: 0.0226 - acc: 0.9929
Epoch 13/20
60000/60000 [==============================] - 24s 400us/step - loss: 0.0213 - acc: 0.9933
Epoch 14/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0214 - acc: 0.9939
Epoch 15/20
60000/60000 [==============================] - 18s 302us/step - loss: 0.0207 - acc: 0.9938
Epoch 16/20
60000/60000 [==============================] - 20s 330us/step - loss: 0.0190 - acc: 0.9942
Epoch 17/20
60000/60000 [==============================] - 20s 332us/step - loss: 0.0174 - acc: 0.9949
Epoch 18/20
60000/60000 [==============================] - 20s 331us/step - loss: 0.0201 - acc: 0.9948
Epoch 19/20
60000/60000 [==============================] - 20s 333us/step - loss: 0.0175 - acc: 0.9949
Epoch 20/20
60000/60000 [==============================] - 19s 321us/step - loss: 0.0177 - acc: 0.9952
10000/10000 [==============================] - 1s 119us/step
TEST LOSS: 0.08804376618057431
TEST ACCURACY: 0.986
9
     
2 ->   32,64


60000/60000 [==============================] - 18s 302us/step - loss: 0.1498 - acc: 0.9545
Epoch 2/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0540 - acc: 0.9832
Epoch 3/20
60000/60000 [==============================] - 44s 741us/step - loss: 0.0433 - acc: 0.9864
Epoch 4/20
60000/60000 [==============================] - 46s 765us/step - loss: 0.0335 - acc: 0.9893
Epoch 5/20
60000/60000 [==============================] - 33s 545us/step - loss: 0.0285 - acc: 0.9906
Epoch 6/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0223 - acc: 0.9929
Epoch 7/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0211 - acc: 0.9933
Epoch 8/20
60000/60000 [==============================] - 20s 339us/step - loss: 0.0203 - acc: 0.9932
Epoch 9/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0185 - acc: 0.9941
Epoch 10/20
60000/60000 [==============================] - 21s 355us/step - loss: 0.0162 - acc: 0.9947
Epoch 11/20
60000/60000 [==============================] - 21s 355us/step - loss: 0.0160 - acc: 0.9950
Epoch 12/20
60000/60000 [==============================] - 22s 367us/step - loss: 0.0128 - acc: 0.9960
Epoch 13/20
60000/60000 [==============================] - 20s 340us/step - loss: 0.0145 - acc: 0.9955
Epoch 14/20
60000/60000 [==============================] - 20s 329us/step - loss: 0.0132 - acc: 0.9957
Epoch 15/20
60000/60000 [==============================] - 19s 321us/step - loss: 0.0135 - acc: 0.9959
Epoch 16/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0114 - acc: 0.9965
Epoch 17/20
60000/60000 [==============================] - 23s 377us/step - loss: 0.0126 - acc: 0.9962
Epoch 18/20
60000/60000 [==============================] - 19s 318us/step - loss: 0.0121 - acc: 0.9963
Epoch 19/20
60000/60000 [==============================] - 20s 326us/step - loss: 0.0110 - acc: 0.9969
Epoch 20/20
60000/60000 [==============================] - 20s 337us/step - loss: 0.0142 - acc: 0.9959
10000/10000 [==============================] - 1s 124us/step
TEST LOSS: 0.053206579726541164
TEST ACCURACY: 0.9906
9

1-> 32

        60000/60000 [==============================] - 25s 416us/step - loss: 0.2055 - acc: 0.9380
Epoch 2/20
60000/60000 [==============================] - 21s 346us/step - loss: 0.0810 - acc: 0.9752
Epoch 3/20
60000/60000 [==============================] - 23s 378us/step - loss: 0.0578 - acc: 0.9816
Epoch 4/20
60000/60000 [==============================] - 21s 351us/step - loss: 0.0439 - acc: 0.9862
Epoch 5/20
60000/60000 [==============================] - 21s 356us/step - loss: 0.0338 - acc: 0.9892
Epoch 6/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0284 - acc: 0.9907
Epoch 7/20
60000/60000 [==============================] - 22s 360us/step - loss: 0.0241 - acc: 0.9918
Epoch 8/20
60000/60000 [==============================] - 21s 354us/step - loss: 0.0217 - acc: 0.9929
Epoch 9/20
60000/60000 [==============================] - 21s 352us/step - loss: 0.0202 - acc: 0.9934
Epoch 10/20
60000/60000 [==============================] - 20s 334us/step - loss: 0.0162 - acc: 0.9944
Epoch 11/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0179 - acc: 0.9941
Epoch 12/20
60000/60000 [==============================] - 21s 353us/step - loss: 0.0149 - acc: 0.9949
Epoch 13/20
60000/60000 [==============================] - 21s 344us/step - loss: 0.0165 - acc: 0.9943
Epoch 14/20
60000/60000 [==============================] - 21s 348us/step - loss: 0.0131 - acc: 0.9956
Epoch 15/20
60000/60000 [==============================] - 21s 347us/step - loss: 0.0135 - acc: 0.9955
Epoch 16/20
60000/60000 [==============================] - 21s 350us/step - loss: 0.0128 - acc: 0.9958
Epoch 17/20
60000/60000 [==============================] - 20s 341us/step - loss: 0.0135 - acc: 0.9957
Epoch 18/20
60000/60000 [==============================] - 21s 343us/step - loss: 0.0127 - acc: 0.9958
Epoch 19/20
60000/60000 [==============================] - 23s 382us/step - loss: 0.0122 - acc: 0.9960
Epoch 20/20
60000/60000 [==============================] - 21s 349us/step - loss: 0.0088 - acc: 0.9969
10000/10000 [==============================] - 1s 109us/step
TEST LOSS: 0.07443016861326614
TEST ACCURACY: 0.9855
'''

>>>>>>> weights removed
