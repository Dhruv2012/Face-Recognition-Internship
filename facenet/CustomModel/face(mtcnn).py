import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import os, time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import glob
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)


detector = MTCNN()
FACE_DIR = "images(face cropped)/"
FACE_DIR1 = "D:\Summer Intern 2019\FACENET/testing/train_mtcnn\P17EC001"
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def main_test():
    detector = MTCNN(steps_threshold=(.05, .7, .7))
    create_folder(FACE_DIR1)
    for file in glob.glob("D:\Summer Intern 2019\FACENET/testing/train\P17EC001\*.jpg"):

        (dirname, filename) = os.path.split(file)
        print(filename)
        img = cv2.imread(dirname + '/' + filename, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(img)
        if (len(faces) == 1):
            print(detector.detect_faces(img))
            bounding_box = faces[0]['box']
            face_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                       bounding_box[0]: bounding_box[0] + bounding_box[2]]
            #face_aligned = face_aligner.align(img, img_gray, bounding_box)
            path = FACE_DIR1 + "/" + (filename)
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(path, face_img)
        elif (len(faces)>1):
            for i in range(len(faces)):
                bounding_box = faces[i]['box']
                face_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
                           bounding_box[0]: bounding_box[0] + bounding_box[2]]
                #face_aligned = face_aligner.align(img, img_gray, bounding_box)
                path = "FACE_DIR1" + "/" + str(i) + (filename)
                face_img = cv2.resize(face_img, (200, 200))
                cv2.imwrite(path, face_img)

    for file1 in glob.glob("D:\Summer Intern 2019\FACENET/testing/train\P17EC001\*.png"):
        (dirname1, filename1) = os.path.split(file1)
        print(filename1)
        img1 = cv2.imread(dirname1 + '/' + filename1, 1)
        img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        faces1 = detector.detect_faces(img)
        if (len(faces1) == 1):
            bounding_box = faces1[0]['box']
            face_img1 = img1[bounding_box[1]:bounding_box[1] + bounding_box[3],
                        bounding_box[0]: bounding_box[0] + bounding_box[2]]
            #face_aligned1 = face_aligner.align(img1, img_gray1, bounding_box)
            path1 = FACE_DIR1 + "/" + filename1
            face_img1 = cv2.resize(face_img1, (200, 200))
            cv2.imwrite(path1, face_img1)
        elif (len(faces1)>1):
            for i in range(len(faces1)):
                bounding_box = faces1[i]['box']
                face_img1 = img1[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]: bounding_box[0] + bounding_box[2]]
                #face_aligned1 = face_aligner.align(img1, img_gray1, bounding_box)
                path1 = FACE_DIR1 + "/" + str(i) + filename1
                face_img1 = cv2.resize(face_img1, (200, 200))
                cv2.imwrite(path1, face_img1)


#main_test()