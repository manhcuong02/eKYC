import torch
import numpy as np

class FaceOrientationDetector():
    """This class detects the orientation of a face in an image."""
    def __init__(self):
        
        self.frontal_range = [25, 60]
        
    
    def calculate_angle(self, v1: (list, tuple, torch.Tensor, np.ndarray), v2: (list, tuple, torch.Tensor, np.ndarray)):
        '''
        Calculate the angle between 2 vectors v1 and v2
        '''
        if isinstance(v1, torch.Tensor):
            v1 = v1.numpy()
        else: 
            v1 = np.array(v1)
        if isinstance(v2, torch.Tensor):
            v2 = v2.numpy()
        else: 
            v2 = np.array(v2)
            
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        rad = np.arccos(cosine)

        degrees = np.degrees(rad)
        
        return np.round(degrees)        
    
    def detect(self, landmarks: np.ndarray):
        '''
            Detects the orientation of a face based on landmarks.

            Parameters:
                landmarks (np.ndarray): A list of 6 points representing the positions on the face [left eye, right eye, nose, left mouth, right mouth].

            Returns:
                tuple: Returns a tuple indicating the face orientations [front: frontal, left: turned left, right: turned right].
        '''
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        
        left2right_eye = right_eye - left_eye
        lefteye2nose = nose - left_eye
        
        left_angle = self.calculate_angle(left2right_eye, lefteye2nose)
        
        right2left_eye = left_eye - right_eye
        righteye2nose = nose - right_eye
        
        right_angle = self.calculate_angle(right2left_eye, righteye2nose)
                
        if self.frontal_range[0] <= left_angle <= self.frontal_range[1] \
            and self.frontal_range[0] <= right_angle <= self.frontal_range[1]:
                return 'front'
        
        # elif  left_angle > self.frontal_range[1] and right_angle > self.frontal_range[1]:
        #     return 'down'
        
        # elif  left_angle < self.frontal_range[0] and right_angle < self.frontal_range[0]:
        #     return 'up'
            
        elif left_angle < right_angle:
            return 'right'
        
        return 'left'