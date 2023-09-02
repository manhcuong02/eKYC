from PyQt5.QtGui import QFont 
from PyQt5.QtWidgets import QPushButton

def add_button(self, title, x, y, w, h, event = None, font_size = 10, mode = 'show', disabled = False):
    assert mode in ['hide', 'show']
    button = QPushButton(self)
    button.setText(title)
    button.move(x,y)
    button.setFixedSize(w,h)
    
    font = QFont()
    font.setPointSize(font_size)
    button.setFont(font)
    
    if mode == 'show':
        button.show()
    else:
        button.hide()
    
    if event:
        button.clicked.connect(event)
        
    button.setDisabled(disabled)
    
    return button