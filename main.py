import sys

import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from face_verification import *
from facenet.models.mtcnn import MTCNN
from gui.page1 import *
from gui.page2 import *
from gui.page3 import *
from gui.utils import *
from liveness_detection.blink_detection import BlinkDetector
from liveness_detection.emotion_prediction import EmotionPredictor
from liveness_detection.face_orientation import FaceOrientationDetector
from utils.functions import *
from verification_models import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  

        self.window_heigt = 800
        self.window_width = 1600
        self.setWindowTitle("eKYC GUI")
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)
        
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.mtcnn = MTCNN(device = self.device)
        
        self.verification_model = VGGFace.load_model(device = self.device)
        
        self.blink_detector = BlinkDetector()
        self.face_orientation_detector = FaceOrientationDetector()
        self.emotion_preidictor = EmotionPredictor(device = self.device)

        # camera
        self.camera = cv.VideoCapture(0)

        # stack widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.first_page = IDCardPhoto(main_window = self)
        self.second_page = VerificationWindow(camera = self.camera, main_window = self)
        self.third_page = ChallengeWindow(camera = self.camera, main_window = self, mtcnn = self.mtcnn, list_models = [self.blink_detector, self.face_orientation_detector, self.emotion_preidictor])

        self.stacked_widget.addWidget(self.first_page)
        self.stacked_widget.addWidget(self.second_page)
        self.stacked_widget.addWidget(self.third_page)
        
    def verify(self):
        id_image = get_image(self.first_page.img_path)
        verification_image = self.second_page.verification_image

        verified = verify(id_image, verification_image, self.mtcnn, self.verification_model, model_name = "VGG-Face1")

        return verified        

    def switch_page(self, index):
        if index == 0:
            self.first_page.clear_window()
            self.second_page.close_camera()
            self.third_page.close_camera()
        
        elif index == 1:
            self.second_page.clear_window()
            self.second_page.open_camera()
            self.third_page.close_camera()
        
        elif index == 2:
            self.third_page.clear_window()
            self.third_page.open_camera()
            self.second_page.close_camera()
            
        self.stacked_widget.setCurrentIndex(index)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
