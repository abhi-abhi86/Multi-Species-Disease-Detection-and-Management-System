# DiseaseDetectionApp/ui/image_search_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton, QApplication
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
import requests

class ImageFetcher(QObject):
    """
    Worker object to fetch an image from a URL in a separate thread.
    """
    finished = pyqtSignal(QPixmap)
    error = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        """Fetches the image and emits a signal with the result."""
        try:
            response = requests.get(self.url, timeout=15)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            pixmap = QPixmap()
            if not pixmap.loadFromData(response.content):
                self.error.emit("Invalid image data received from the URL.")
                return

            self.finished.emit(pixmap)

        except requests.exceptions.Timeout:
            self.error.emit("The request timed out. The server is taking too long to respond.")
        except requests.exceptions.HTTPError as http_err:
            self.error.emit(f"HTTP error occurred: {http_err.response.status_code} {http_err.response.reason}")
        except requests.exceptions.RequestException as e:
            self.error.emit(f"A network error occurred.\nPlease check your connection.\nDetails: {e}")
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")


class ImageSearchDialog(QDialog):
    """
    A dialog that allows users to search for and view images of diseases
    from the database, with non-blocking image loading.
    """
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.worker_thread = None
        self.current_pixmap = None

        self.setWindowTitle("Search Disease Image")
        self.setMinimumSize(400, 400)

        self.layout = QVBoxLayout(self)

        self.disease_combo = QComboBox()
        self.disease_combo.addItems(["Select a disease..."] + [d['name'] for d in self.database])
        self.disease_combo.currentIndexChanged.connect(self.start_image_fetch)

        self.image_label = QLabel("Select a disease to see its image.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 8px;")

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        self.layout.addWidget(self.disease_combo)
        self.layout.addWidget(self.image_label, 1)
        self.layout.addWidget(self.close_button)

    def start_image_fetch(self, index):
        """
        Initiates the image fetching process in a background thread.
        """
        self.current_pixmap = None # Reset current pixmap
        self.image_label.setPixmap(QPixmap()) # Clear previous image

        if index == 0:
            self.image_label.setText("Select a disease to see its image.")
            return

        disease_name = self.disease_combo.currentText()
        disease_info = next((d for d in self.database if d['name'] == disease_name), None)

        if not (disease_info and disease_info.get('image_url')):
            self.image_label.setText(f"No image URL found for {disease_name}.")
            return

        url = disease_info['image_url']
        self.image_label.setText("Loading image...")
        self.set_controls_enabled(False)

        # Setup and start the worker thread
        self.worker_thread = QThread()
        fetcher = ImageFetcher(url)
        fetcher.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(fetcher.run)
        fetcher.finished.connect(self.on_image_loaded)
        fetcher.error.connect(self.on_fetch_error)

        # Clean up the thread once it's done
        fetcher.finished.connect(self.worker_thread.quit)
        fetcher.error.connect(self.worker_thread.quit)
        fetcher.finished.connect(fetcher.deleteLater)
        fetcher.error.connect(fetcher.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_image_loaded(self, pixmap):
        """Handles the successful download of an image."""
        self.current_pixmap = pixmap
        self.image_label.setText("")
        self.display_scaled_image()
        self.set_controls_enabled(True)

    def on_fetch_error(self, error_message):
        """Displays an error message if the image fails to load."""
        disease_name = self.disease_combo.currentText()
        self.image_label.setText(f"Failed to load image for {disease_name}.\n\nReason: {error_message}")
        self.set_controls_enabled(True)

    def set_controls_enabled(self, enabled):
        """Enables or disables UI controls during fetching."""
        self.disease_combo.setEnabled(enabled)
        self.close_button.setEnabled(enabled)

    def display_scaled_image(self):
        """Scales and displays the current pixmap."""
        if self.current_pixmap:
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
        """Ensures the worker thread is terminated if the dialog is closed."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait() # Wait for the thread to finish
        event.accept()

