
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class GeocoderWorker(QObject):
    """Worker to geocode locations in the background."""
    location_found = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, diagnosis_locations):
        super().__init__()
        self.diagnosis_locations = diagnosis_locations
        self.is_running = True

    def run(self):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="disease_detection_app")
        except ImportError:
            self.error.emit("Geocoding library not found. Please install 'geopy'.")
            return

        if not self.diagnosis_locations:
            self.location_found.emit("No locations have been recorded yet.")
            self.finished.emit()
            return

        for item in self.diagnosis_locations:
            if not self.is_running:
                break
            try:
                location = geolocator.geocode(item['location'])
                if location:
                    result = (f"• Disease: {item['disease']}\n"
                              f"  Location: {item['location']}\n"
                              f"  Coordinates: ({location.latitude:.4f}, {location.longitude:.4f})\n")
                    self.location_found.emit(result)
                else:
                    self.location_found.emit(f"• Could not find coordinates for: {item['location']}\n")
            except Exception as e:
                self.location_found.emit(f"• Error geocoding {item['location']}: {e}\n")
        self.finished.emit()

    def stop(self):
        self.is_running = False

class MapDialog(QDialog):
    def __init__(self, diagnosis_locations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recorded Disease Locations")
        self.setMinimumSize(500, 400)
        self.layout = QVBoxLayout(self)

        self.results_view = QTextEdit()
        self.results_view.setReadOnly(True)
        self.results_view.setPlaceholderText("Fetching location data...")

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        self.layout.addWidget(self.results_view)
        self.layout.addWidget(self.close_button)

        self.thread = QThread()
        self.worker = GeocoderWorker(diagnosis_locations)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.location_found.connect(self.results_view.append)
        self.worker.error.connect(self.show_error)

        self.thread.start()

    def show_error(self, message):
        self.results_view.setText(f"<p style='color:red;'>{message}</p>")

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)
