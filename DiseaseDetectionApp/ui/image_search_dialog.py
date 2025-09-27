# DiseaseDetectionApp/ui/image_search_dialog.py
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class ImageSearchDialog(QDialog):
    """
    A dialog that allows users to search for and view images of diseases
    from the local database.
    """

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.current_pixmap = None

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
        self.layout.addWidget(self.image_label, 1)
        self.layout.addWidget(self.close_button)

    def display_image(self, index):
        """
        Loads and displays the selected disease image from a local path.
        """
        self.current_pixmap = None
        self.image_label.setPixmap(QPixmap())  # Clear previous image

        if index == 0:
            self.image_label.setText("Select a disease to see its image.")
            return

        disease_name = self.disease_combo.currentText()
        disease_info = next((d for d in self.database if d['name'] == disease_name), None)

        if not (disease_info and disease_info.get('image_url')):
            self.image_label.setText(f"No image path found for {disease_name}.")
            return

        # Construct the full path to the image relative to the 'data' directory
        # This makes the path independent of where the main script is run from
        try:
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
            image_path = os.path.join(base_path, disease_info['image_url'])

            if not os.path.exists(image_path):
                self.image_label.setText(f"Image file not found.\nExpected at: {image_path}")
                print(f"Debug: Failed to find image at {image_path}")
                return

            self.current_pixmap = QPixmap(image_path)
            self.image_label.setText("")
            self.display_scaled_image()
        
        except Exception as e:
            self.image_label.setText(f"An error occurred while loading the image:\n{e}")
            print(f"Debug: Error loading image - {e}")


    def display_scaled_image(self):
        """Scales and displays the current pixmap."""
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.image_label.setPixmap(self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        """Handles window resize to rescale the image."""
        super().resizeEvent(event)
        self.display_scaled_image()
