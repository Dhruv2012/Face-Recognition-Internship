import tensorflow as tf
#from tf import ker
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

datagen = ImageDataGenerator(
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #shear_range=0.0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant')

PATH = 'D:/FACENET/testing/train_alignfix/P17EC017/P17EC017_047.jpg'
img = tf.keras.preprocessing.image.load_img(PATH)  
x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x,save_to_dir='D:/FACENET/testing/train_alignfix/P17EC017', save_prefix='1', save_format='jpg'):
    i += 1
    if i > 10:
        break 
