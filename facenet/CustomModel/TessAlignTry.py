import numpy as np
import cv2
import matplotlib.pyplot as plt
from align import *
import dlib
alignment = AlignDlib('D:\Summer Intern 2019\FACENET\CustomModel\shape_predictor_5_face_landmarks.dat')
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

aligned  = align_image('D:\Summer Intern 2019\FACENET\CustomModel\images/tilted.jpg')
print (aligned)
#cv2.imshow(aligned,'aligned')
#cv2.imwrite('D:\Summer Intern 2019\FACENET\CustomModel\images(face cropped)' , aligned)
#plt.imshow(aligned)