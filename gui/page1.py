from PyQt5.QtCore import QPoint, QRect, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QStackedWidget,
                             QVBoxLayout, QWidget)

import cv2 as cv
from .utils import *


class IDCardPhoto(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.window_heigt = 800
        self.window_width = 1600
                
        # Thiết lập tiêu đề và kích thước của cửa sổ
        self.setWindowTitle('Choose ID Card Photo')
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)
        
        self.font = QFont()
        self.font.setPointSize(13)
        self.font.setFamily("Times New Roman")

        self.label = QLabel(self)
        self.label.setText('Please select the front side of your national identity card.')
        self.label.move(550, 100)
        self.label.setFont(self.font)
        
        self.exit_button = add_button(self, "Exit", 800, 700, 150, 50, exit)
    
        self.select_image_button = add_button(self, "Select ID Card", 320, 700, 150, 50, self.selectImage)
        self.next = add_button(self, "Next", 1280, 700, 150, 50, self.switch_page, disabled = True)
        self.in_image = QLabel(self)
        
        self.img_path = None
    
    def switch_page(self):
        self.main_window.switch_page(1)     
        
    def rescale_image(self, width, height):
        return int(width * 400 / height), 400

    def selectImage(self):
        # Hiển thị hộp thoại chọn tệp ảnh và lấy tên tệp ảnh được chọn
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)')
        
        if file_name:
            self.img_path = file_name
            # Tải ảnh từ tệp và hiển thị nó trên QLabel
            pixmap = QPixmap(file_name)
            img = cv.imread(file_name)
            width, height = self.rescale_image(img.shape[1], img.shape[0])
            self.in_image.setGeometry(QRect(800 - width //2 , 150, width, height))
            self.in_image.setPixmap(pixmap.scaled(width, height))  
            self.in_image.show()  
            self.next.setDisabled(False)

    def clear_window(self):
        self.in_image.hide()