import random

import cv2 as cv
import numpy as np
import torch

from facenet.models.mtcnn import MTCNN
from liveness_detection.emotion_prediction import *
from utils.functions import extract_face
from utils.plot import *

model = MTCNN()

video = cv.VideoCapture(0)

emotion_predictor = EmotionPredictor()

while True:
    ret, frame = video.read()
    if ret:
        frame = cv.flip(frame, 1)
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        face, box, landmarks = extract_face(rgb_img, model, padding = 0)
        
        if box is not None:
            emotion = emotion_predictor.predict(face)
            cv.putText(frame, emotion, (20,20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
            
        cv.imshow("", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
