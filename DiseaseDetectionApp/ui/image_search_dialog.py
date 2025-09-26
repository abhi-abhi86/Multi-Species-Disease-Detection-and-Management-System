# DiseaseDetectionApp/ui/image_search_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton, QMessageBox, QApplication
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import requests

class ImageSearchDialog(QDialog):
    """
    A dialog that allows users to search for and view images of diseases
    from the database.
    """
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("Search Disease Image")
        self.setMinimumSize(400, 400)

        self.layout = QVBoxLayout(self)

        self.disease_combo = QComboBox()
        self.disease_combo.addItems(["Select a disease..."] + [d['name'] for d in self.database])
        self.disease_combo.currentIndexChanged.connect(self.display_image)

        self.image_label = QLabel("Select a disease to see its image.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 8px;")

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        self.layout.addWidget(self.disease_combo)
        self.layout.addWidget(self.image_label, 1) # Give it stretch factor
        self.layout.addWidget(self.close_button)

    def display_image(self, index):
        """
        Fetches and displays the image for the selected disease.
        """
        if index == 0:
            self.image_label.setText("Select a disease to see its image.")
            self.image_label.setPixmap(QPixmap()) # Clear pixmap
            return

        disease_name = self.disease_combo.currentText()
        disease_info = next((d for d in self.database if d['name'] == disease_name), None)

        if disease_info and 'image_url' in disease_info and disease_info['image_url']:
            url = disease_info['image_url']
            self.image_label.setText("Loading image...")
            QApplication.processEvents()
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                pixmap = QPixmap()
                pixmap.loadFromData(response.content)

                if not pixmap.isNull():
                    self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    self.image_label.setText("")
                else:
                    self.image_label.setText(f"Could not load image for {disease_name}.\nURL may be invalid or the image format is not supported.")

            except requests.exceptions.RequestException as e:
                self.image_label.setText(f"Failed to download image for {disease_name}.\nError: {e}")
        else:
            self.image_label.setText(f"No image URL found for {disease_name}.")
            self.image_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        """
        Handles window resize events to rescale the displayed image.
        """
        super().resizeEvent(event)
        # Rescale pixmap when window is resized
        if self.disease_combo.currentIndex() > 0:
            # Re-trigger image display to scale it correctly
            current_pixmap = self.image_label.pixmap()
            if current_pixmap and not current_pixmap.isNull():
                 self.image_label.setPixmap(current_pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
