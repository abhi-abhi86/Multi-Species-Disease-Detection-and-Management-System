# DiseaseDetectionApp/ui/image_search_dialog.py
import requests
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal

from core.google_search import search_google_images

class ImageFetchWorker(QObject):
    """
    Worker to fetch an image from a URL in the background.
    """
    finished = pyqtSignal(bytes)
    error = pyqtSignal(str)

    def __init__(self, disease_name):
        super().__init__()
        self.disease_name = disease_name

    def run(self):
        try:
            # Search for the image URL
            image_url = search_google_images(self.disease_name)
            
            if not image_url:
                self.error.emit(f"No image found online for '{self.disease_name}'.")
                return

            # Fetch the image data
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            self.finished.emit(response.content)

        except requests.exceptions.RequestException as e:
            self.error.emit(f"Network error: {e}")
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")


class ImageSearchDialog(QDialog):
    """
    A dialog that allows users to search for and view images of diseases
    by fetching them from Google Images.
    """
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.current_pixmap = None
        self.worker_thread = None
        self.worker = None

        self.setWindowTitle("Search Disease Image Online")
        self.setMinimumSize(400, 400)

        self.layout = QVBoxLayout(self)

        self.disease_combo = QComboBox()
        self.disease_combo.addItems(["Select a disease..."] + [d['name'] for d in self.database])
        self.disease_combo.currentIndexChanged.connect(self.start_image_search)

        self.image_label = QLabel("Select a disease to search for its image.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 8px;")

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        self.layout.addWidget(self.disease_combo)
        self.layout.addWidget(self.image_label, 1)
        self.layout.addWidget(self.close_button)

    def start_image_search(self, index):
        """
        Initiates the image search in a background thread.
        """
        self.current_pixmap = None
        self.image_label.setPixmap(QPixmap())

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()

        if index == 0:
            self.image_label.setText("Select a disease to search for its image.")
            return

        disease_name = self.disease_combo.currentText()
        self.image_label.setText(f"Searching for images of '{disease_name}'...")

        self.worker_thread = QThread()
        self.worker = ImageFetchWorker(disease_name)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_search_finished)
        self.worker.error.connect(self.on_search_error)
        
        # Clean up the thread
        self.worker.finished.connect(self.cleanup_thread)
        self.worker.error.connect(self.cleanup_thread)
        
        self.worker_thread.start()

    def on_search_finished(self, image_data):
        self.current_pixmap = QPixmap()
        self.current_pixmap.loadFromData(image_data)
        
        if self.current_pixmap.isNull():
            self.image_label.setText("Failed to load the downloaded image data.")
        else:
            self.image_label.setText("")  # Clear loading text
            self.display_scaled_image()

    def on_search_error(self, error_message):
        self.image_label.setText(f"Search Failed:\n{error_message}")

    def cleanup_thread(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

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
    
    def closeEvent(self, event):
        """Ensures the worker thread is stopped when the dialog is closed."""
        self.cleanup_thread()
        super().closeEvent(event)
