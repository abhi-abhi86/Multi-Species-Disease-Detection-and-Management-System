import sys
import os
from PySide6.QtWidgets import QApplication, QMessageBox





WATERMARK_AUTHOR = "abhi-abhi86"
WATERMARK_CHECK = True

def check_watermark():
    """Check if watermark is intact. Application will fail if removed."""
    if not WATERMARK_CHECK or WATERMARK_AUTHOR != "abhi-abhi86":
        print("ERROR: Watermark protection violated. Application cannot start.")
        print("Made by: abhi-abhi86")
        sys.exit(1)


check_watermark()





sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DiseaseDetectionApp.ui.main_window import MainWindow


def main():
    """
    The main entry point for the application.
    Initializes the QApplication and the main window.
    """
    app = QApplication(sys.argv)


    app.setApplicationName("Multi-Species Disease Detection and Management System")
    app.setOrganizationName("AI-Diagnosis-System")

    window = MainWindow()


    if not window.ml_processor or not window.ml_processor.model:
        QMessageBox.critical(
            None,
            "Critical Error: Model Not Found",
            "The custom AI model (`disease_model.pt`) or its class map (`class_to_name.json`) could not be found.\n\n"
            "Please run the `train_disease_classifier.py` script from your terminal to train the model before starting the application.\n\n"
            "The application will now close."
        )
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
