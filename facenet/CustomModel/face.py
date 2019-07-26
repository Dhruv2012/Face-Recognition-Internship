
import cv2
import numpy as np
import os, time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import glob
from align import *


detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
alignment = AlignDlib('/home/ml/FACENET/CustomModel/shape_predictor_68_face_landmarks.dat')

FACE_DIR = "images(face cropped)/"
src0 = "/home/ml/FACENET/CustomModel/images"
dest0 = "/home/ml/FACENET/CustomModel/images(face cropped)"
######################## TRAIN ##############################


src1 = "/home/ml/FACENET/testing/train/P17EC001"
src2 = "/home/ml/FACENET/testing/train/P17EC002"
src3 = "/home/ml/FACENET/testing/train/P17EC003"
src4 = "/home/ml/FACENET/testing/train/P17EC004"
src5 = "/home/ml/FACENET/testing/train/P17EC005"
src6 = "/home/ml/FACENET/testing/train/P17EC006"
src7 = "/home/ml/FACENET/testing/train/P17EC007"
src8 = "/home/ml/FACENET/testing/train/P17EC008"
src9 = "/home/ml/FACENET/testing/train/P17EC009"
src10 = "/home/ml/FACENET/testing/train/P17EC010"
src11 = "/home/ml/FACENET/testing/train/P17EC011"
src12 = "/home/ml/FACENET/testing/train/P17EC012"
src13 = "/home/ml/FACENET/testing/train/P17EC013"
src14 = "/home/ml/FACENET/testing/train/P17EC014"
src15 = "/home/ml/FACENET/testing/train/P17EC015"
src17 = "/home/ml/FACENET/testing/train/P17EC017"

FACE_DIR1 = "/home/ml/FACENET/testing/train_alignedv1/P17EC001"
FACE_DIR2 = "/home/ml/FACENET/testing/train_alignedv1/P17EC002"
FACE_DIR3 = "/home/ml/FACENET/testing/train_alignedv1/P17EC003"
FACE_DIR4 = "/home/ml/FACENET/testing/train_alignedv1/P17EC004"
FACE_DIR5 = "/home/ml/FACENET/testing/train_alignedv1/P17EC005"
FACE_DIR6 = "/home/ml/FACENET/testing/train_alignedv1/P17EC006"
FACE_DIR7 = "/home/ml/FACENET/testing/train_alignedv1/P17EC007"
FACE_DIR8 = "/home/ml/FACENET/testing/train_alignedv1/P17EC008"
FACE_DIR9 = "/home/ml/FACENET/testing/train_alignedv1/P17EC009"
FACE_DIR10 = "/home/ml/FACENET/testing/train_alignedv1/P17EC010"
FACE_DIR11 = "/home/ml/FACENET/testing/train_alignedv1/P17EC011"
FACE_DIR12 = "/home/ml/FACENET/testing/train_alignedv1/P17EC012"
FACE_DIR13 = "/home/ml/FACENET/testing/train_alignedv1/P17EC013"
FACE_DIR14 = "/home/ml/FACENET/testing/train_alignedv1/P17EC014"
FACE_DIR15 = "/home/ml/FACENET/testing/train_alignedv1/P17EC015"
FACE_DIR17 = "/home/ml/FACENET/testing/train_alignedv1/P17EC017"


################## TEST ######################################

test_1 = "/home/ml/FACENET/testing/test/P17EC001"
test_2 = "/home/ml/FACENET/testing/test/P17EC002"
test_3 = "/home/ml/FACENET/testing/test/P17EC003"
test_4 = "/home/ml/FACENET/testing/test/P17EC004"
test_5 = "/home/ml/FACENET/testing/test/P17EC005"
test_6 = "/home/ml/FACENET/testing/test/P17EC006"
test_7 = "/home/ml/FACENET/testing/test/P17EC007"
test_8 = "/home/ml/FACENET/testing/test/P17EC008"
test_9 = "/home/ml/FACENET/testing/test/P17EC009"
test_10 = "/home/ml/FACENET/testing/test/P17EC010"
test_11 = "/home/ml/FACENET/testing/test/P17EC011"
test_12 = "/home/ml/FACENET/testing/test/P17EC012"
test_13 = "/home/ml/FACENET/testing/test/P17EC013"
test_14 = "/home/ml/FACENET/testing/test/P17EC014"
test_15 = "/home/ml/FACENET/testing/test/P17EC015"
test_17 = "/home/ml/FACENET/testing/test/P17EC017"



d3 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Aman"
d4 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Ankit"
d5 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Chirayu"
d10 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Harnish"
d11 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Lokesh"
d12 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Mansi"
d13 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/meet"
d14 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Milie"
d15 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/dataset/Vats"

'''
e3 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Aman"
e4 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Ankit"
e5 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Chirayu"
e10 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Harnish"
e11 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Lokesh"
e12 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Mansi"
e13 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/meet"
e14 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Milie"
e15 = "/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Vats"
'''

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
x=0
y=0
w=0
h=0

def align_mod(img, bb):
    return alignment.align(96, img, bb,
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def cropalign():
    create_folder(FACE_DIR)                 #For single image per class
    for file in glob.glob("/home/ml/FACENET/CustomModel/images/*.jpg"):
        (dirname, filename) = os.path.split(file)
        print(filename)
        img = cv2.imread(dirname + '/' +  filename, 1)
        img = cv2.resize(img, (96, 96))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray, 2)
        print(len(faces))
        for i in faces:
            x = i.left()
            y = i.top()
            w = i.right() - x
            h = i.bottom() - y
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            face_aligned = face_aligner.align(img, img_gray, i)
            face_img = face_aligned
            print(face_img.shape)
            cv2.imshow("aligned", face_img)
            path = dest + "/" + (filename)
            cv2.imwrite(path, face_img)



    for file1 in glob.glob("images/*.png"):
        (dirname1, filename1) = os.path.split(file1)
        print(filename1)
        img1 = cv2.imread(dirname1 + '/' + filename1, 1)
        img1 = cv2.resize(img1, (96, 96))
        img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces1 = detector(img_gray1, 2)
        print(len(faces1))
        for i in faces1:
            x = i.left()
            y = i.top()
            w = i.right() - x
            h = i.bottom() - y
            #cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
            face_aligned1 = face_aligner.align(img1, img_gray1, i)
            face_img1 = face_aligned1
            print(face_img1.shape)
            cv2.imshow("aligned", face_img1)
            path1 = dest + "/" + filename1
            cv2.imwrite(path1, face_img1)

    for file2 in glob.glob("/home/ml/FACENET/CustomModel/images/test_image/*.jpg"):
        (dirname2, filename2) = os.path.split(file2)
        print(filename2)
        img2 = cv2.imread(dirname2 + '/' + filename2, 1)
        img2 = cv2.resize(img2, (96, 96))
        img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces2 = detector(img_gray2, 2)
        print(len(faces2))
        for i in faces2:
            x = i.left()
            y = i.top()
            w = i.right() - x
            h = i.bottom() - y
            #cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)
            face_aligned2 = face_aligner.align(img2, img_gray2, i)
            face_img2 = face_aligned2
            print(face_img2.shape)
            cv2.imshow("aligned", face_img2)
            path2 = "/home/ml/FACENET/CustomModel/images(face cropped)/test_image" + "/" + filename2
            cv2.imwrite(path2, face_img2)


def cropalign_cnn(src,dest,upscale):
    create_folder(dest)  # For single image per class
    for file in glob.glob(src + "/*.jpg" ):
        (dirname, filename) = os.path.split(file)
        print(filename)
        img = cv2.imread(dirname + '/' + filename, 1)
        #img = cv2.resize(img, (128, 128))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cnn_detector(img_gray, upscale)
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
            face_aligned = face_aligner.align(img, img_gray, i.rect)
            face_img = face_aligned
            print(face_img.shape)
            cv2.imshow("aligned", face_img)
            face_img = cv2.resize(face_img, (96, 96))
            if len(faces) ==1:
                path = dest + "/" + (filename)
            elif len(faces)>1:
                path = dest + "/" + str(j) + (filename)
            cv2.imwrite(path, face_img)

    for file1 in glob.glob(src + "/*.png"):
        (dirname1, filename1) = os.path.split(file1)
        print(filename1)
        img1 = cv2.imread(dirname1 + '/' + filename1, 1)
        #img1 = cv2.resize(img1, (128, 128))
        img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces1 = cnn_detector(img_gray1, upscale)
        print(len(faces1))
        j = 0
        for i in faces1:
            j = j + 1
            x = i.rect.left()
            y = i.rect.top()
            w = i.rect.right() - x
            h = i.rect.bottom() - y
            # cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
            face_aligned1 = face_aligner.align(img1, img_gray1, i.rect)
            face_img1 = face_aligned1
            print(face_img1.shape)
            cv2.imshow("aligned", face_img1)
            face_img1 = cv2.resize(face_img1, (96, 96))
            if len(faces1) ==1:
                path1 = dest + "/" + (filename1)
            elif len(faces1)>1:
                path1 = dest + "/" + str(j) + (filename1)
            cv2.imwrite(path1, face_img1)





def cropalign_cnn_alignfix(src,dest,upscale):
    create_folder(dest)  # For single image per class
    for file in glob.glob(src + "/*.jpg" ):
        (dirname, filename) = os.path.split(file)
        print(filename)
        img = cv2.imread(dirname + '/' + filename, 1)
        j =0
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cnn_detector(img_gray, upscale)
        print(len(faces))
        for i in faces:
            j = j + 1
            x = i.rect.left()
            y = i.rect.top()
            w = i.rect.right() - x
            h = i.rect.bottom() - y
            face_img = align_mod(img,i.rect)
            print(face_img.shape)
            cv2.imshow("aligned", face_img)
            face_img = cv2.resize(face_img, (96, 96))
            if len(faces) ==1:
                path = dest + "/" + (filename)
            elif len(faces)>1:
                path = dest + "/" + str(j) + (filename)
            cv2.imwrite(path, face_img)
            j= j+ 1

    for file1 in glob.glob(src + "/*.png"):
        (dirname1, filename1) = os.path.split(file1)
        print(filename1)
        img1 = cv2.imread(dirname1 + '/' + filename1, 1)
        j = 0
        img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces1 = cnn_detector(img_gray1, upscale)
        print(len(faces1))
        for i in faces1:
            j = j + 1
            x = i.rect.left()
            y = i.rect.top()
            w = i.rect.right() - x
            h = i.rect.bottom() - y
            face_img1 = align_mod(img1,rect)
            print(face_img1.shape)
            cv2.imshow("aligned", face_img1)
            face_img1 = cv2.resize(face_img1, (96, 96))
            if len(faces) ==1:
                path1 = dest + "/" + (filename1)
            elif len(faces1)>1:
                path1 = dest + "/" + str(j) + (filename1)
            cv2.imwrite(path1, face_img1)
            j = j + 1
#cropalign_cnn_alignfix(d3,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Aman",1)
#cropalign_cnn_alignfix(d4,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Ankit",1)
#cropalign_cnn_alignfix(d5,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Chirayu",1)
#cropalign_cnn_alignfix(d10,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Harnish",1)
#cropalign_cnn_alignfix(d11,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Lokesh",1)
#cropalign_cnn_alignfix(d12,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Mansi",1)
#cropalign_cnn_alignfix(d13,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/meet",1)
cropalign_cnn_alignfix(d14,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Milie",1)
cropalign_cnn_alignfix(d15,"/home/ml/FACENET/Face_Recognition_dataset/DRISHTI/Cropped/Vats",1)
'''
cropalign_cnn_alignfix(test_1,"/home/ml/FACENET/testing/test_alignfix/P17EC001",1)
cropalign_cnn_alignfix(test_2,"/home/ml/FACENET/testing/test_alignfix/P17EC002",1)
cropalign_cnn_alignfix(test_3,"/home/ml/FACENET/testing/test_alignfix/P17EC003",1)
cropalign_cnn_alignfix(test_4,"/home/ml/FACENET/testing/test_alignfix/P17EC004",1)
cropalign_cnn_alignfix(test_5,"/home/ml/FACENET/testing/test_alignfix/P17EC005",1)
cropalign_cnn_alignfix(test_6,"/home/ml/FACENET/testing/test_alignfix/P17EC006",1)
cropalign_cnn_alignfix(test_7,"/home/ml/FACENET/testing/test_alignfix/P17EC007",1)
cropalign_cnn_alignfix(test_8,"/home/ml/FACENET/testing/test_alignfix/P17EC008",1)
cropalign_cnn_alignfix(test_9,"/home/ml/FACENET/testing/test_alignfix/P17EC009",1)
cropalign_cnn_alignfix(test_10,"/home/ml/FACENET/testing/test_alignfix/P17EC010",1)
cropalign_cnn_alignfix(test_11,"/home/ml/FACENET/testing/test_alignfix/P17EC011",1)
cropalign_cnn_alignfix(test_12,"/home/ml/FACENET/testing/test_alignfix/P17EC012",1)
cropalign_cnn_alignfix(test_13,"/home/ml/FACENET/testing/test_alignfix/P17EC013",1)
cropalign_cnn_alignfix(test_14,"/home/ml/FACENET/testing/test_alignfix/P17EC014",1)
cropalign_cnn_alignfix(test_15,"/home/ml/FACENET/testing/test_alignfix/P17EC015",1)
cropalign_cnn_alignfix(test_17,"/home/ml/FACENET/testing/test_alignfix/P17EC017",1)


cropalign_cnn_alignfix(src1,"/home/ml/FACENET/testing/train_alignfix/P17EC001",1)
cropalign_cnn_alignfix(src2,"/home/ml/FACENET/testing/train_alignfix/P17EC002",1)
cropalign_cnn_alignfix(src3,"/home/ml/FACENET/testing/train_alignfix/P17EC003",1)
cropalign_cnn_alignfix(src4,"/home/ml/FACENET/testing/train_alignfix/P17EC004",1)
cropalign_cnn_alignfix(src5,"/home/ml/FACENET/testing/train_alignfix/P17EC005",1)
cropalign_cnn_alignfix(src6,"/home/ml/FACENET/testing/train_alignfix/P17EC006",1)
cropalign_cnn_alignfix(src7,"/home/ml/FACENET/testing/train_alignfix/P17EC007",1)
cropalign_cnn_alignfix(src8,"/home/ml/FACENET/testing/train_alignfix/P17EC008",1)
cropalign_cnn_alignfix(src9,"/home/ml/FACENET/testing/train_alignfix/P17EC009",1)
cropalign_cnn_alignfix(src10,"/home/ml/FACENET/testing/train_alignfix/P17EC010",1)
cropalign_cnn_alignfix(src11,"/home/ml/FACENET/testing/train_alignfix/P17EC011",1)
cropalign_cnn_alignfix(src12,"/home/ml/FACENET/testing/train_alignfix/P17EC012",1)
cropalign_cnn_alignfix(src13,"/home/ml/FACENET/testing/train_alignfix/P17EC013",1)
cropalign_cnn_alignfix(src14,"/home/ml/FACENET/testing/train_alignfix/P17EC014",1)
cropalign_cnn_alignfix(src15,"/home/ml/FACENET/testing/train_alignfix/P17EC015",1)
cropalign_cnn_alignfix(src17,"/home/ml/FACENET/testing/train_alignfix/P17EC017",1)
'''

















#cropalign_cnn(src0,dest0,1)
'''
cropalign_cnn(test_1,"/home/ml/FACENET/testing/test_alignedv1/P17EC001",1)
cropalign_cnn(test_2,"/home/ml/FACENET/testing/test_alignedv1/P17EC002",1)
cropalign_cnn(test_3,"/home/ml/FACENET/testing/test_alignedv1/P17EC003",1)
cropalign_cnn(test_4,"/home/ml/FACENET/testing/test_alignedv1/P17EC004",1)
cropalign_cnn(test_5,"/home/ml/FACENET/testing/test_alignedv1/P17EC005",1)
cropalign_cnn(test_6,"/home/ml/FACENET/testing/test_alignedv1/P17EC006",1)
cropalign_cnn(test_7,"/home/ml/FACENET/testing/test_alignedv1/P17EC007",1)
cropalign_cnn(test_8,"/home/ml/FACENET/testing/test_alignedv1/P17EC008",1)
cropalign_cnn(test_9,"/home/ml/FACENET/testing/test_alignedv1/P17EC009",1)
cropalign_cnn(test_10,"/home/ml/FACENET/testing/test_alignedv1/P17EC010",1)
cropalign_cnn(test_11,"/home/ml/FACENET/testing/test_alignedv1/P17EC011",1)
cropalign_cnn(test_12,"/home/ml/FACENET/testing/test_alignedv1/P17EC012",1)
cropalign_cnn(test_13,"/home/ml/FACENET/testing/test_alignedv1/P17EC013",1)
cropalign_cnn(test_14,"/home/ml/FACENET/testing/test_alignedv1/P17EC014",1)
cropalign_cnn(test_15,"/home/ml/FACENET/testing/test_alignedv1/P17EC015",1)
cropalign_cnn(test_17,"/home/ml/FACENET/testing/test_alignedv1/P17EC017",1)

'''
'''
cropalign_cnn(src1,FACE_DIR1,1)
cropalign_cnn(src2,FACE_DIR2,1)
cropalign_cnn(src3,FACE_DIR3,1)
cropalign_cnn(src4,FACE_DIR4,1)
cropalign_cnn(src5,FACE_DIR5,1)
cropalign_cnn(src6,FACE_DIR6,1)
cropalign_cnn(src7,FACE_DIR7,1)
cropalign_cnn(src8,FACE_DIR8,1)
cropalign_cnn(src9,FACE_DIR9,1)
cropalign_cnn(src10,FACE_DIR10,1)
cropalign_cnn(src11,FACE_DIR11,1)
cropalign_cnn(src12,FACE_DIR12,1)
cropalign_cnn(src13,FACE_DIR13,1)
cropalign_cnn(src14,FACE_DIR14,1)
cropalign_cnn(src15,FACE_DIR15,1)
cropalign_cnn(src17,FACE_DIR17,1)
'''





#################################   MTCNN ###################################################33
