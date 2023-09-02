import dlib
import cv2 as cv
import numpy as np
from imutils import face_utils
import os
import torch
import math

class BlinkDetector():
    '''A class for detecting eye blinking in facial images'''
    
    def __init__(self):
        # cargar modelo para deteccion de puntos de ojos
        landmark_path = os.path.join(os.path.dirname(__file__), 'landmarks/shape_predictor_68_face_landmarks.dat')
        self.predictor_eyes = dlib.shape_predictor(landmark_path)

        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        self.counter = 0
        self.total = 0

    def eye_blink(self, rgb_image: np.ndarray, rect : (np.ndarray, torch.Tensor, list, tuple, dlib.rectangle), thresh = 1):
        '''
        Detects eye blinking in a given face region of an input BGR image.

        Parameters:
        - rgb_image (np.ndarray): Input RGB image as a numpy array.
        - rect: A bounding rectangle [x1, y1, x2, y2] defining the face region.
        - thresh (int): A challenge-response threshold that the user needs to surpass.

        Returns:
        - out (bool): True if the user successfully surpasses the challenge (>= thresh), False otherwise (< thresh).
        '''
        if isinstance(rect, torch.Tensor):
            rect = dlib.rectangle(*rect.long())
        elif isinstance(rect, (np.ndarray, list, tuple)):
            rect = np.array(rect).astype(np.uint32)
            rect = dlib.rectangle(*rect)

        gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
    
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor_eyes(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        
        
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # check to see if the eye aspect ratio is below the blink threshold
        # and if so, increment the blink frame counter
        if ear < self.EYE_AR_THRESH:
            self.counter += 1
            
        # otherwise, the eye aspect ratio is not below the blink threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                self.total += 1
            # reset the eye frame counter
            self.counter = 0
        
        # leftEyeHull = cv.convexHull(leftEye)
        # rightEyeHull = cv.convexHull(rightEye)

        # cv.drawContours(bgr_image, [leftEyeHull], -1, (0, 255, 0), 2)
        # cv.drawContours(bgr_image, [rightEyeHull], -1, (0, 255, 0), 2)
        # cv.putText(bgr_image, f"total: {self.total}", (20,20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        # cv.putText(bgr_image, f"EAR: {ear :.2f}", (20,60), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        # return bgr_image    
    
        if self.total >= thresh:
            self.total = 0
            return True
        return False

    def eye_aspect_ratio(self,eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = math.dist(eye[1], eye[5])
        B = math.dist(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = math.dist(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear
    
    
if __name__ == '__main__':
    blink_detector = BlinkDetector()
    
    