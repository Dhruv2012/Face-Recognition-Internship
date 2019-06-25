import numpy as np
import cv2
import matplotlib.pyplot as plt
from align import *
import dlib
import os, time
from imutils import face_utils
from imutils.face_utils import FaceAligner
import glob


detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
alignment = AlignDlib('D:\Summer Intern 2019\FACENET\CustomModel\shape_predictor_68_face_landmarks.dat')


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

img = cv2.imread('D:\Summer Intern 2019\FACENET/testing/test\P17EC011/P17EC011_0043.jpg', 1)
#img = cv2.resize(img, (96, 96))
#img = cv2.imread('D:\Summer Intern 2019\FACENET\CustomModel\images/testtt.jpg', 1)
'''
aligned  = align_image(img)
print (aligned)
cv2.imwrite('D:\Summer Intern 2019\FACENET\CustomModel\images(face cropped)/aligned3.jpg' , aligned)
'''
dest = 'D:\Summer Intern 2019\FACENET\CustomModel\images(face cropped)'

def align_mod(img, bb):
    return alignment.align(96, img, bb,
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cnn_detector(img_gray, 1)
print(len(faces))
j =0
for i in faces:
    j = j +1
    x = i.rect.left()
    y = i.rect.top()
    w = i.rect.right() - x
    h = i.rect.bottom() - y
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    print(i.rect)
    aligned4 = align_mod(img,i.rect)
    path = dest + "/" + str(j) + "aligned5.jpg"
    cv2.imwrite(path,aligned4)