from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import QTimer
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Khai báo biến
        self.frame_count = 0

        # Tạo QLabel
        self.label = QLabel(self)
        self.label.move(50,50)
        self.label.adjustSize()
        # Khởi tạo QTimer
        self.timer = QTimer()

        # Thiết lập khe hở cho QTimer
        self.timer.setInterval(30)  # 30 fps

        # Thiết lập khe hở cho kết nối tới slot
        self.timer.timeout.connect(self.update_frame)

        self.update_label_size()  
    
        # Bắt đầu timer
        self.timer.start()

    def update_label_size(self):
        self.label.adjustSize()

    def update_frame(self):
        # Tăng biến đếm
        self.frame_count += 1

        # Kiểm tra nếu biến đếm đạt giá trị nhất định
        if self.frame_count == 100:
            # Đặt văn bản cho nhãn
            self.label.setText("Văn bản mới")
            self.update_label_size()  
            
import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
a = MyWidget()
a.show()
sys.exit(app.exec_())