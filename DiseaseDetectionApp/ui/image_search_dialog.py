# DiseaseDetectionApp/ui/image_search_dialog.py
import requests
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal

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
            # Append "disease" to the query to get more relevant results
            image_url = search_google_images(self.disease_name + " disease")
            
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
    A redesigned dialog that allows users to search for disease images online
    using a more intuitive text input and a dedicated search button.
    """
    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.worker_thread = None
        self.worker = None

        self.setWindowTitle("Search Disease Image Online")
        self.setMinimumSize(450, 450)

        self.layout = QVBoxLayout(self)

        # --- NEW: Intuitive Search Bar Layout ---
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type a disease name to search...")
        self.search_input.returnPressed.connect(self.start_image_search) # Allow pressing Enter to search
        self.search_input.setStyleSheet("padding: 5px; border-radius: 5px;")

        self.search_button = QPushButton("Search")
        self.search_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.search_button.clicked.connect(self.start_image_search)
        self.search_button.setStyleSheet(
            "padding: 5px 15px; background-color: #4f8cff; color: white; border-radius: 5px;"
        )
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        # --- End of New Layout ---

        self.image_label = QLabel("Enter a disease name above and click 'Search'.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0; border-radius: 8px;")

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        self.layout.addLayout(search_layout) # Add the new search bar layout
        self.layout.addWidget(self.image_label, 1) # The '1' makes the image label expand
        self.layout.addWidget(self.close_button)

    def start_image_search(self):
        """
        Initiates the image search in a background thread based on the text input.
        """
        disease_name = self.search_input.text().strip()
        if not disease_name:
            self.image_label.setText("Please enter a disease name to search.")
            return

        # Clear previous results and disable search button
        self.current_pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(f"Searching for images of '{disease_name}'...")
        self.search_button.setEnabled(False)

        # Ensure any previous worker is stopped
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()

        # Set up and start the new background worker
        self.worker_thread = QThread()
        self.worker = ImageFetchWorker(disease_name)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_search_finished)
        self.worker.error.connect(self.on_search_error)
        
        # Connect signals for cleanup
        self.worker.finished.connect(self.cleanup_thread)
        self.worker.error.connect(self.cleanup_thread)
        
        self.worker_thread.start()

    def on_search_finished(self, image_data):
        """Handles the successful download of image data."""
        self.current_pixmap = QPixmap()
        self.current_pixmap.loadFromData(image_data)
        
        if self.current_pixmap.isNull():
            self.image_label.setText("Failed to load the downloaded image data.")
        else:
            self.image_label.setText("")  # Clear "Searching..." text
            self.display_scaled_image()

    def on_search_error(self, error_message):
        """Displays an error message if the search fails."""
        self.image_label.setText(f"Search Failed:\n{error_message}")

    def cleanup_thread(self):
        """Re-enables the UI and cleans up the worker thread."""
        self.search_button.setEnabled(True)
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    def display_scaled_image(self):
        """Scales and displays the current pixmap in the label."""
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

