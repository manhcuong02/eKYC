import cv2 as cv
import numpy as np
import torch
import random

from facenet.models.mtcnn import MTCNN
from liveness_detection.face_orientation import *
from utils.plot import *

model = MTCNN()

video = cv.VideoCapture(0)

orientation_detector = FaceOrientationDetector()

mode = ['front', 'left', 'right']

while True:
    ret, frame = video.read()
    if ret:
        frame = cv.flip(frame, 1)
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxes, _,landmarks = model.detect(rgb_img, landmarks = True)
        if boxes is not None:
            orientation = orientation_detector.detect(landmarks[0])
            
            frame = plot_landmarks_mtcnn(frame, landmarks[0], orientation = orientation)
        cv.imshow("", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
