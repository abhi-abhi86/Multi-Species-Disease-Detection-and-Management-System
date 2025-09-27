# New robust, animated, and stable PyQt6 GUI code

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QPropertyAnimation, QRect

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Disease Detection App")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label = QLabel("Welcome to the Disease Detection App!")
        self.layout.addWidget(self.label)

        self.button = QPushButton("Start Detection")
        self.button.clicked.connect(self.start_detection)
        self.layout.addWidget(self.button)

    def start_detection(self):
        self.label.setText("Detection in progress...")
        self.animate_button()

    def animate_button(self):
        animation = QPropertyAnimation(self.button, b"geometry")
        animation.setDuration(1000)
        animation.setStartValue(QRect(100, 100, 200, 50))
        animation.setEndValue(QRect(300, 100, 200, 50))
        animation.setLoopCount(-1)  # Infinite loop
        animation.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())