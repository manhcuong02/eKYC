import cv2 as cv
import numpy as np
import torch

from facenet.models.mtcnn import MTCNN
from liveness_detection.blink_detection import *

model = MTCNN()

video = cv.VideoCapture("videos/eye_blink.mov")

blink_detector = BlinkDetector()

while True:
    ret, frame = video.read()
    if ret:
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxes, _, = model.detect(rgb_img)
        if boxes is not None: 
            verified = blink_detector.eye_blink(frame, boxes[0], 3)
        if verified:
            print(1)
        cv.imshow("", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
