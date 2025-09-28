# DiseaseDetectionApp/ui/main_window.py
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar,
    QLineEdit, QProgressBar, QStatusBar, QDialog
)
from PyQt6.QtGui import QPixmap, QAction, QFont, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import os

# --- Import local modules ---
# This structure makes dependencies clearer.
from ui.add_disease_dialog import AddNewDiseaseDialog
from ui.chatbot_dialog import ChatbotDialog
from ui.image_search_dialog import ImageSearchDialog
from ui.map_dialog import MapDialog
from core.data_handler import load_database, save_disease
from core.ml_processor import MLProcessor
from core.worker import DiagnosisWorker
from core.report_generator import generate_pdf_report

class CustomDropLabel(QLabel):
    """ A QLabel widget that accepts drag-and-drop for image files. """
    fileDropped = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop an Image Here\nor Click 'Upload Image'")
        self.setFont(QFont("Arial", 11))
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #007bff;
                background-color: #f8f9fa;
                border-radius: 10px;
                color: #007bff;
            }
        """)
        self.setMinimumHeight(180)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Check if any of the dropped files are valid images
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    event.acceptProposedAction()
                    self.setStyleSheet("border: 2px solid #0056b3; background-color: #e9ecef; border-radius: 10px; color: #0056b3;")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #007bff; background-color: #f8f9fa; border-radius: 10px; color: #007bff;")

    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #007bff; background-color: #f8f9fa; border-radius: 10px; color: #007bff;")
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.fileDropped.emit(url.toLocalFile())
                    event.acceptProposedAction()
                    return

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Species Disease Detection and Management System")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("background-color: #fdfdff;")

        # --- Application State ---
        self.database = load_database()
        self.ml_processor = MLProcessor()
        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.diagnosis_locations = []
        self.worker_thread = None
        self.diagnosis_worker = None
        
        # This is the base directory of the application (DiseaseDetectionApp)
        self.base_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # --- Main UI Setup ---
        self.setup_menu()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        self.domain_tabs = {}
        for domain, label in [("Plant", "🌱 Plants"), ("Human", "🧑 Humans"), ("Animal", "🐾 Animals")]:
            tab = self.create_domain_tab(domain)
            self.tab_widget.addTab(tab, label)
            self.domain_tabs[domain] = tab

    def setup_menu(self):
        menubar = self.menuBar()
        # File Menu
        file_menu = menubar.addMenu("File")
        add_action = QAction("Add New Disease...", self)
        add_action.triggered.connect(self.open_add_disease_dialog)
        file_menu.addAction(add_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        # Tools Menu
        tools_menu = menubar.addMenu("Tools")
        chatbot_action = QAction("Disease Chatbot...", self)
        chatbot_action.triggered.connect(self.open_chatbot)
        tools_menu.addAction(chatbot_action)
        search_image_action = QAction("Search Disease Image Online...", self)
        search_image_action.triggered.connect(self.open_image_search_dialog)
        tools_menu.addAction(search_image_action)
        map_action = QAction("View Disease Map...", self)
        map_action.triggered.connect(self.open_map_dialog)
        tools_menu.addAction(map_action)

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Input Group ---
        input_group = QGroupBox("1. Provide Input")
        input_layout = QGridLayout(input_group)
        
        image_label = CustomDropLabel()
        image_label.fileDropped.connect(lambda path: self.set_image(path, domain_name))
        
        upload_btn = QPushButton("Upload Image")
        upload_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        symptom_input = QTextEdit()
        symptom_input.setPlaceholderText("Or, describe the symptoms here...")
        
        location_input = QLineEdit()
        location_input.setPlaceholderText("Optional: Enter location (e.g., New Delhi, India)")

        diagnose_btn = QPushButton("Diagnose")
        diagnose_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        diagnose_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setTextVisible(False)

        input_layout.addWidget(image_label, 0, 0, 3, 1)
        input_layout.addWidget(upload_btn, 3, 0)
        input_layout.addWidget(QLabel("Symptoms:"), 0, 1)
        input_layout.addWidget(symptom_input, 1, 1, 1, 2)
        input_layout.addWidget(QLabel("Location:"), 2, 1)
        input_layout.addWidget(location_input, 2, 2)
        input_layout.addWidget(diagnose_btn, 3, 1, 1, 2)
        
        main_layout.addWidget(input_group)
        main_layout.addWidget(progress_bar)

        # --- Result Group ---
        result_group = QGroupBox("2. Diagnosis Result")
        result_layout = QGridLayout(result_group)

        reference_image_label = QLabel("Reference image will appear here.")
        reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_image_label.setFixedSize(250, 250)
        reference_image_label.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; background-color: #f8f9fa;")

        result_display = QTextEdit()
        result_display.setReadOnly(True)
        
        pdf_button = QPushButton("Save Report as PDF")
        pdf_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        pdf_button.setEnabled(False) # Disabled until a diagnosis is made

        result_layout.addWidget(reference_image_label, 0, 0)
        result_layout.addWidget(result_display, 0, 1)
        result_layout.addWidget(pdf_button, 1, 0, 1, 2)
        result_layout.setColumnStretch(1, 1)

        main_layout.addWidget(result_group)

        # --- Store widgets for easy access ---
        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
        main_widget.reference_image_label = reference_image_label
        main_widget.location_input = location_input
        main_widget.diagnose_btn = diagnose_btn
        main_widget.progress_bar = progress_bar
        main_widget.pdf_button = pdf_button
        main_widget.diagnosis_data = None # To store the latest result for the PDF report

        # --- Connect signals ---
        upload_btn.clicked.connect(lambda: self.upload_image(domain_name))
        diagnose_btn.clicked.connect(lambda: self.run_diagnosis(domain_name))
        pdf_button.clicked.connect(lambda: self.save_report_as_pdf(domain_name))

        return main_widget

    def set_image(self, file_path, domain):
        if file_path:
            tab = self.domain_tabs[domain]
            self.current_image_paths[domain] = file_path
            pixmap = QPixmap(file_path)
            tab.image_label.setPixmap(pixmap.scaled(
                tab.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            # Clear other inputs when a new image is set
            tab.symptom_input.clear()
            tab.result_display.clear()
            tab.pdf_button.setEnabled(False)

    def upload_image(self, domain):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_image(file_path, domain)

    def run_diagnosis(self, domain):
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()

        # Input validation
        if not image_path and not symptoms:
            QMessageBox.warning(self, "Input Missing", "Please upload an image OR describe the symptoms to proceed.")
            return

        # Prioritize image if both are provided
        if image_path and symptoms:
            print("Both image and symptoms provided. Prioritizing image for AI diagnosis.")
            tab.symptom_input.clear()
            symptoms = ""

        # --- Start Diagnosis Thread ---
        tab.diagnose_btn.setEnabled(False)
        tab.pdf_button.setEnabled(False)
        tab.progress_bar.setVisible(True)
        tab.progress_bar.setRange(0, 0) # Indeterminate progress bar
        tab.result_display.setPlainText("Starting diagnosis...")

        self.worker_thread = QThread()
        self.diagnosis_worker = DiagnosisWorker(
            self.ml_processor,
            image_path,
            symptoms,
            domain,
            self.database
        )
        self.diagnosis_worker.moveToThread(self.worker_thread)
        
        # Connect signals from worker to UI slots
        self.worker_thread.started.connect(self.diagnosis_worker.run)
        self.diagnosis_worker.finished.connect(self.on_diagnosis_complete)
        self.diagnosis_worker.error.connect(self.on_diagnosis_error)
        self.diagnosis_worker.progress.connect(lambda msg: tab.result_display.setPlainText(msg))
        
        # Cleanup connection
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def on_diagnosis_complete(self, result, confidence, wiki_summary, predicted_stage, pubmed_summary, domain):
        self.cleanup_worker_and_ui(domain)
        tab = self.domain_tabs[domain]
        
        # Store result for PDF generation
        tab.diagnosis_data = {**result, 'confidence': confidence, 'stage': predicted_stage, 'image_path': self.current_image_paths[domain]}

        stages_str = "\n".join([f"  • <b>{k}:</b> {v}" for k, v in result.get("stages", {}).items()])
        
        output_html = (
            f"<h3>Diagnosis: {result['name']}</h3>"
            f"<b>Confidence Score:</b> <span style='color:#28a745; font-weight:bold;'>{confidence:.1f}%</span><br>"
            f"<b>Predicted Stage:</b> {predicted_stage}<br><br>"
            f"<b>Description:</b><br>{result.get('description', 'N/A')}<br><br>"
            f"<b>Wikipedia Summary:</b><br>{wiki_summary}<br><br>"
            f"<b>Known Stages:</b><br>{stages_str if stages_str else 'N/A'}<br><br>"
            f"<b style='color:#17a2b8;'>Solution/Cure:</b><br><span style='color:#17a2b8;'>{result.get('solution', 'N/A')}</span><br><br>"
            f"<b>Recent Research (PubMed):</b><br>{pubmed_summary}"
        )
        
        tab.result_display.setHtml(output_html)
        tab.pdf_button.setEnabled(True)

        # --- Display Reference Image ---
        # Standardizing on 'image_url' for consistency
        relative_image_path = result.get('image_url')
        if relative_image_path:
            # Construct the full path from the base application directory
            full_image_path = os.path.join(self.base_app_dir, relative_image_path)
            if os.path.exists(full_image_path):
                pixmap = QPixmap(full_image_path)
                tab.reference_image_label.setPixmap(pixmap.scaled(
                    tab.reference_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                print(f"Reference image not found at: {full_image_path}")
                tab.reference_image_label.setText(f"Image not found:\n{relative_image_path}")
        else:
            tab.reference_image_label.setText("No reference image\nin database.")
        
        # --- Log Location ---
        location = tab.location_input.text().strip()
        if location:
            self.diagnosis_locations.append({"disease": result['name'], "location": location})
            self.status_bar.showMessage(f"Diagnosis complete. Location '{location}' logged.", 5000)
            tab.location_input.clear()
        else:
            self.status_bar.showMessage("Diagnosis complete.", 3000)
            
    def on_diagnosis_error(self, error_message, domain):
        self.cleanup_worker_and_ui(domain)
        tab = self.domain_tabs[domain]
        tab.result_display.setHtml(f"<h3 style='color:red;'>Diagnosis Failed</h3><p>{error_message}</p>")
        self.status_bar.showMessage("Diagnosis failed.", 4000)

    def cleanup_worker_and_ui(self, domain):
        """Resets the UI and stops the worker thread."""
        tab = self.domain_tabs[domain]
        tab.progress_bar.setVisible(False)
        tab.diagnose_btn.setEnabled(True)
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker_thread = None

    def open_add_disease_dialog(self):
        dialog = AddNewDiseaseDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not all([data['name'], data['description'], data['solution']]):
                QMessageBox.warning(self, "Incomplete Data", "Please fill in at least the Name, Description, and Solution fields.")
                return
            
            success, error_msg = save_disease(data)
            if success:
                self.database = load_database() # Reload database to include new entry
                QMessageBox.information(self, "Success", f"Disease '{data['name']}' has been saved successfully.")
            else:
                QMessageBox.critical(self, "Save Error", f"Failed to save the disease:\n{error_msg}")

    def save_report_as_pdf(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Data", "Please run a diagnosis before saving a report.")
            return

        # Sanitize filename
        safe_name = re.sub(r'[\s/\\:*?"<>|]+', '_', tab.diagnosis_data['name']).lower()
        default_path = f"{safe_name}_report.pdf"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", default_path, "PDF Files (*.pdf)")
        
        if file_path:
            success, error_msg = generate_pdf_report(file_path, tab.diagnosis_data)
            if success:
                QMessageBox.information(self, "Success", f"Report saved successfully to:\n{file_path}")
            else:
                QMessageBox.critical(self, "PDF Error", f"Failed to generate PDF report:\n{error_msg}")

    def open_chatbot(self):
        dialog = ChatbotDialog(self.database, self)
        dialog.exec()

    def open_image_search_dialog(self):
        dialog = ImageSearchDialog(self.database, self)
        dialog.exec()

    def open_map_dialog(self):
        dialog = MapDialog(self.diagnosis_locations, self)
        dialog.exec()

    def closeEvent(self, event):
        # Ensure the worker thread is stopped cleanly when closing the app
        if self.worker_thread and self.worker_thread.isRunning():
            self.diagnosis_worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()
