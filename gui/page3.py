import cv2 as cv
from PyQt5.QtCore import QRect, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel

from .utils import *
from challenge_response import *

class ChallengeWindow(QDialog):
    def __init__(self, camera, main_window, mtcnn, list_models = [], parent=None):
        super().__init__(parent)
        
        self.main_window = main_window
        
        self.window_heigt = 800
        self.window_width = 1600
                
        # Thiết lập tiêu đề và kích thước của cửa sổ
        self.setWindowTitle('Challenge response')
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)

        self.font = QFont()
        self.font.setPointSize(13)
        self.font.setFamily("Times New Roman")

        self.label = QLabel(self)
        self.label.setText('Verify your authenticity by completing the following challenges.')
        self.label.move(520, 100)
        self.label.setFont(self.font)
        
        self.challenge_label = QLabel(self)
        self.challenge_label.setFont(self.font)
        self.challenge_label.move(550, 650)
        self.update_challenge_label()
        
        self.camera_label = QLabel(self)

        self.camera = camera  # Open the default camera (usually the built-in webcam)
        self.timer = QTimer(self)
        
        # button 
        self.next = add_button(self, "Exit", 1280, 700, 150, 50, exit)
        self.back = add_button(self, "Back", 320, 700, 150, 50, self.back_switch_page)
        
        #  models
        self.list_models = list_models
        self.mtcnn = mtcnn
        self.count_frame = 0
        self.isCorrect = False
        self.count_correct = 0 # đếm số câu hỏi đúng 
        self.count_delay_frame = 0 # 
        self.challenge, self.question = get_challenge_and_question()        
        
    def rescale_image(self):
        return 640, 480
    
    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv.flip(frame, 1)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            width, height = self.rescale_image()
            self.camera_label.setGeometry(QRect(800 - width //2 , 150, width, height))
            self.camera_label.setPixmap(pixmap)
            
            # sau 100 frame thì hiện câu hỏi 
            if self.count_delay_frame < 100:
                self.count_delay_frame += 1
            else:
                if self.isCorrect == False and self.count_correct < 3:
                    self.isCorrect = result_challenge_response(frame, self.challenge, self.question, self.list_models, self.mtcnn)
                    self.update_challenge_label(question = self.question)
                    
                elif self.isCorrect == True and self.count_correct < 3:
                    self.update_challenge_label(text = "<font color = green>Correct!</font>")
                    self.count_frame += 1

                    if self.count_frame == 100:
                        self.count_correct += 1        
                        self.count_frame = 0
                        if self.count_correct == 3:
                            self.update_challenge_label(text = "<font color = green>You have successfully established your identity!</font>", coordinates = (600, 650))
                        else:
                            self.challenge, self.question = get_challenge_and_question()
                            self.update_challenge_label(question = self.question)
                            self.isCorrect = False
                        
    def back_switch_page(self):
        self.main_window.switch_page(1)  

    def closeEvent(self, event):
        self.camera.release()
        self.timer.stop()
        event.accept()
        
    def open_camera(self):
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)
        
    def close_camera(self):
        self.timer.stop()
        
    def clear_window(self):
        self.challenge_label.hide() 
        
        self.count_frame = 0
        self.isCorrect = False
        self.count_correct = 0
        self.count_delay_frame = 0
        
        self.challenge, self.question = get_challenge_and_question()
    
    def update_challenge_label(self, text = None, question = None, coordinates = None):
        assert  not (text is not None and question is not None)
        
        if text is not None:
            self.challenge_label.setText(text)
            self.challenge_label.move(750, 650)
            
        if question is not None:
            if isinstance(self.question, str):
                text = f"Question {self.count_correct + 1}/3: {self.question}"
            else:
                text = f"Question {self.count_correct + 1}/3: {self.question[0]}"
            self.challenge_label.setText(text)
            self.challenge_label.move(580, 650)
        
        if coordinates is not None:
            self.challenge_label.move(coordinates[0], coordinates[1])
        self.challenge_label.adjustSize()
        self.challenge_label.show()    
        