
import cv2 as cv
from PyQt5.QtCore import QRect, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel

from .utils import *


class VerificationWindow(QDialog):
    def __init__(self, camera, main_window, parent=None):
        super().__init__(parent)
        
        self.main_window = main_window
        
        # set window size
        self.window_heigt = 800
        self.window_width = 1600
        self.setWindowTitle('Verification')
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)

        # font
        self.font = QFont()
        self.font.setPointSize(13)
        self.font.setFamily("Times New Roman")

        # title
        self.label = QLabel(self)
        self.label.setText("Please keep your face in front of the camera and come closer to the camera.")
        self.label.move(470, 100)
        self.label.setFont(self.font)
        
        # process label
        self.process_label = QLabel(self)
        self.process_label.move(730, 650)
        self.process_label.setFont(self.font)
        self.update_process_label()
        
        self.count_frame = 0
        
        # camera label
        self.camera_label = QLabel(self)
        self.camera = camera # Open the default camera (usually the built-in webcam)
        self.timer = QTimer(self)
        
        # button
        self.next = add_button(self, "Next", 1280, 700, 150, 50, self.next_switch_page, disabled= True)
        self.back = add_button(self, "Back", 320, 700, 150, 50, self.back_switch_page)
        self.exit = add_button(self, "Exit", 800, 700, 150, 50, exit)

        # trạng thái sau khi được xác 
        self.verified = False
        
        self.verification_image = None
        
    def update_process_label(self, text = None):
        if text is not None:
            self.process_label.setText(text)
        self.process_label.adjustSize()
        self.process_label.show()
        
    def rescale_image(self):
        return 640, 480
    
    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            self.count_frame += 1
            frame = cv.flip(frame, 1)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            width, height = self.rescale_image()
            self.camera_label.setGeometry(QRect(800 - width //2 , 150, width, height))
            self.camera_label.setPixmap(pixmap)
            
            if self.count_frame == 50:
                self.update_process_label(text = "Verifying...")
                self.verification_image = frame
                self.verified = self.main_window.verify()
                if self.verified == False:
                    self.count_frame = 0
                    self.update_process_label(text = "<font color = red>Verification failed!</font>")
                else:
                    self.update_process_label(text = "<font color = green>Verification successful!</font>")
                    self.next.setDisabled(False)

    def next_switch_page(self):
        self.main_window.switch_page(2)  
        
    def back_switch_page(self):
        self.main_window.switch_page(0)  

    def open_camera(self):
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # Update every 30 milliseconds

    def close_camera(self):
        self.timer.stop()

    def closeEvent(self, event):
        self.camera.release()
        self.timer.stop()
        event.accept()
    
    def clear_window(self):
        self.process_label.hide()
        self.count_frame = 0