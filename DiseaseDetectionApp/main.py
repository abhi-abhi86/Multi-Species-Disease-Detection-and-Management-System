import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox

# --- Path Fix ---
# This adds the project's root directory (the one containing the 'DiseaseDetectionApp' folder)
# to the Python path. This is crucial for making the absolute imports work correctly
# when you run this script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DiseaseDetectionApp.ui.main_window import MainWindow


def main():
    """
    The main entry point for the application.
    Initializes the QApplication and the main window.
    """
    app = QApplication(sys.argv)

    # It's good practice to set application-level metadata
    app.setApplicationName("Multi-Species Disease Detection and Management System")
    app.setOrganizationName("AI-Diagnosis-System")

    window = MainWindow()

    # Check if the AI model was loaded successfully. If not, inform the user.
    if not window.ml_processor or not window.ml_processor.model:
        QMessageBox.critical(
            None,
            "Critical Error: Model Not Found",
            "The custom AI model (`disease_model.pt`) or its class map (`class_to_name.json`) could not be found.\n\n"
            "Please run the `train_disease_classifier.py` script from your terminal to train the model before starting the application.\n\n"
            "The application will now close."
        )
        sys.exit(1)  # Exit if the core component is missing

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
