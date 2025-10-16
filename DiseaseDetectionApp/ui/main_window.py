# DiseaseDetectionApp/ui/main_window.py
import os
import re
import sys
import traceback
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar,
    QLineEdit, QStatusBar, QDialog, QHBoxLayout, QGraphicsOpacityEffect,
    QFormLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QFont, QCursor, QMovie
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QTimer, QSettings

# --- Import local modules ---
from .add_disease_dialog import AddNewDiseaseDialog
from .chatbot_dialog import ChatbotDialog
from .image_search_dialog import ImageSearchDialog
from .map_dialog import MapDialog
from ..core.data_handler import load_database, save_disease
from ..core.ml_processor import MLProcessor
from ..core.worker import DiagnosisWorker
from ..core.report_generator import generate_pdf_report


def show_exception_box(exc_type, exc_value, exc_tb):
    err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    dlg = QMessageBox(QMessageBox.Icon.Critical, "Unexpected Error", f"An unexpected error occurred:\n\n{exc_value}")
    dlg.setDetailedText(err_msg)
    dlg.exec()


sys.excepthook = show_exception_box


class SettingsDialog(QDialog):
    """A dialog for user preferences/settings, using QSettings for persistence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.layout = QFormLayout(self)
        self.settings = QSettings("DiseaseDetectionApp", "UserPreferences")
        # Default folders
        self.image_folder = QLineEdit(self.settings.value("image_folder", os.path.expanduser("~")))
        self.pdf_folder = QLineEdit(self.settings.value("pdf_folder", os.path.expanduser("~")))
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(self.settings.value("theme", "Light"))
        self.layout.addRow("Default Image Folder:", self.image_folder)
        self.layout.addRow("Default PDF Save Folder:", self.pdf_folder)
        self.layout.addRow("Theme:", self.theme_combo)
        # Buttons
        btns = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        self.layout.addRow(btns)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def accept(self):
        self.settings.setValue("image_folder", self.image_folder.text())
        self.settings.setValue("pdf_folder", self.pdf_folder.text())
        self.settings.setValue("theme", self.theme_combo.currentText())
        super().accept()

    @staticmethod
    def get_settings(parent=None):
        dlg = SettingsDialog(parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = QSettings("DiseaseDetectionApp", "UserPreferences")
            return {
                "image_folder": s.value("image_folder", os.path.expanduser("~")),
                "pdf_folder": s.value("pdf_folder", os.path.expanduser("~")),
                "theme": s.value("theme", "Light")
            }
        return None


class AnimatedButton(QPushButton):
    def __init__(self, text, primary_color="#007bff", hover_color="#0056b3"):
        super().__init__(text)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._primary_color = primary_color
        self._hover_color = hover_color
        self._setup_style(self._primary_color)

    def _setup_style(self, color):
        self.setStyleSheet(
            f'QPushButton {{'
            f'  background-color: {color};'
            f'  color: white;'
            f'  border: none;'
            f'  padding: 12px 20px;'
            f'  border-radius: 8px;'
            f'  font-weight: bold;'
            f'  font-family: Arial, sans-serif;'
            f'  font-size: 14px;'
            f'}}'
        )

    def enterEvent(self, event):
        self._setup_style(self._hover_color)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._setup_style(self._primary_color)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._setup_style("#003366")
        QTimer.singleShot(100,
                          lambda: self._setup_style(self._hover_color if self.underMouse() else self._primary_color))
        super().mousePressEvent(event)


class SpinnerLabel(QLabel):
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spinner_path = os.path.join(current_dir, "spinner.gif")
        if os.path.exists(spinner_path):
            movie = QMovie(spinner_path)
            self.setMovie(movie)
        else:
            self.setText("...")
        self.setVisible(False)

    def start(self):
        self.setVisible(True)
        if self.movie():
            self.movie().start()

    def stop(self):
        if self.movie():
            self.movie().stop()
        self.setVisible(False)


class CustomDropLabel(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop an Image Here\nor Click 'Upload Image'")
        self.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.base_style = "border: 2px dashed #3498db; background-color: #ecf0f1; border-radius: 15px; color: #2c3e50; padding: 20px;"
        self.hover_style = "border: 2px solid #2980b9; background-color: #d5dbdb; border-radius: 15px; color: #1c2833; padding: 20px;"
        self.setStyleSheet(self.base_style)
        self.setMinimumHeight(180)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    event.acceptProposedAction()
                    self.setStyleSheet(self.hover_style)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.base_style)

    def dropEvent(self, event):
        self.setStyleSheet(self.base_style)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.fileDropped.emit(url.toLocalFile())
                    event.acceptProposedAction()
                    return


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("DiseaseDetectionApp", "UserPreferences")
        self.apply_theme(self.settings.value("theme", "Light"))
        self.setWindowTitle("Multi-Species Disease Detection and Management System")
        self.setGeometry(100, 100, 1000, 800)
        self.database = load_database()
        self.ml_processor = MLProcessor()
        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.diagnosis_locations = []
        self.worker_thread = None
        self.diagnosis_worker = None
        self.base_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.result_animation = None
        self.setup_menu()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.domain_tabs = {}
        for domain, label in [("Plant", "Plants"), ("Human", "Humans"), ("Animal", "Animals")]:
            tab = self.create_domain_tab(domain)
            self.tab_widget.addTab(tab, label)
            self.domain_tabs[domain] = tab

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        add_action = QAction("Add New Disease...", self)
        add_action.triggered.connect(self.open_add_disease_dialog)
        file_menu.addAction(add_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
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
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        tools_menu.addAction(settings_action)

    def apply_theme(self, theme):
        if theme == "Dark":
            self.setStyleSheet(
                'QMainWindow { background-color: #2c3e50; color: #ecf0f1; font-family: Arial, sans-serif; }'
                'QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #34495e; border-radius: 12px; margin-top: 15px; padding: 10px; color: #ecf0f1; background-color: #34495e; }'
                'QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 5px 15px; background-color: #34495e; color: #ecf0f1; border-radius: 8px; }'
                'QLabel, QLineEdit, QTextEdit { color: #ecf0f1; font-family: Arial, sans-serif; }'
                'QPushButton { font-family: Arial, sans-serif; }'
                'QTabWidget::pane { border: 1px solid #34495e; background-color: #2c3e50; }'
                'QTabBar::tab { background-color: #34495e; color: #ecf0f1; padding: 10px; border-radius: 8px 8px 0 0; margin-right: 2px; font-family: Arial, sans-serif; }'
                'QTabBar::tab:selected { background-color: #3498db; color: #ffffff; }'
                'QTabBar::tab:hover { background-color: #2980b9; }'
            )
        else:
            self.setStyleSheet(
                'QMainWindow { background-color: #ecf0f1; color: #2c3e50; font-family: Arial, sans-serif; }'
                'QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #bdc3c7; border-radius: 12px; margin-top: 15px; padding: 10px; color: #2c3e50; background-color: #ffffff; }'
                'QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 5px 15px; background-color: #3498db; color: #ffffff; border-radius: 8px; }'
                'QLabel, QLineEdit, QTextEdit { color: #2c3e50; font-family: Arial, sans-serif; }'
                'QPushButton { font-family: Arial, sans-serif; }'
                'QTabWidget::pane { border: 1px solid #bdc3c7; background-color: #ecf0f1; }'
                'QTabBar::tab { background-color: #bdc3c7; color: #2c3e50; padding: 10px; border-radius: 8px 8px 0 0; margin-right: 2px; font-family: Arial, sans-serif; }'
                'QTabBar::tab:selected { background-color: #3498db; color: #ffffff; }'
                'QTabBar::tab:hover { background-color: #2980b9; color: #ffffff; }'
            )

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        input_group = QGroupBox("1. Provide Input")
        input_layout = QGridLayout(input_group)
        image_label = CustomDropLabel()
        image_label.fileDropped.connect(lambda path: self.set_image(path, domain_name))
        upload_btn = AnimatedButton("Upload Image")
        symptom_input = QTextEdit()
        symptom_input.setPlaceholderText("Or, describe the symptoms here...")
        location_input = QLineEdit()
        location_input.setPlaceholderText("Optional: Enter location (e.g., New Delhi, India)")
        diagnose_btn = AnimatedButton("Diagnose", primary_color="#28a745", hover_color="#218838")
        cancel_btn = AnimatedButton("Cancel", primary_color="#dc3545", hover_color="#c82333")
        cancel_btn.setVisible(False)
        spinner = SpinnerLabel()
        spinner.setFixedSize(32, 32)
        # Image preview metadata display
        preview_meta = QLabel("")
        preview_meta.setWordWrap(True)
        preview_meta.setStyleSheet("font-size:11px; color:#666;")
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(diagnose_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(spinner)
        input_layout.addWidget(image_label, 0, 0, 4, 1)
        input_layout.addWidget(upload_btn, 4, 0)
        input_layout.addWidget(preview_meta, 5, 0)
        input_layout.addWidget(QLabel("Symptoms:"), 0, 1)
        input_layout.addWidget(symptom_input, 1, 1, 1, 2)
        input_layout.addWidget(QLabel("Location:"), 2, 1)
        input_layout.addWidget(location_input, 2, 2)
        input_layout.addWidget(button_container, 3, 1, 1, 2)
        input_layout.setRowStretch(6, 1)  # Add stretch to make layout more spacious
        main_layout.addWidget(input_group)
        result_group = QGroupBox("2. Diagnosis Result")
        result_layout = QGridLayout(result_group)
        opacity_effect = QGraphicsOpacityEffect(result_group)
        result_group.setGraphicsEffect(opacity_effect)
        result_group.setVisible(False)
        reference_image_label = QLabel("Reference image will appear here.")
        reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_image_label.setFixedSize(300, 300)
        reference_image_label.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #ffffff; padding: 5px;")
        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #ffffff; padding: 10px; font-family: Arial, sans-serif; font-size: 14px;")
        pdf_button = AnimatedButton("Save Report as PDF")
        pdf_button.setEnabled(False)
        result_layout.addWidget(reference_image_label, 0, 0)
        result_layout.addWidget(result_display, 0, 1)
        result_layout.addWidget(pdf_button, 1, 0, 1, 2)
        result_layout.setColumnStretch(1, 1)
        main_layout.addWidget(result_group)
        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
        main_widget.reference_image_label = reference_image_label
        main_widget.location_input = location_input
        main_widget.diagnose_btn = diagnose_btn
        main_widget.cancel_btn = cancel_btn
        main_widget.pdf_button = pdf_button
        main_widget.spinner = spinner
        main_widget.diagnosis_data = None
        main_widget.result_group = result_group
        main_widget.result_opacity_effect = opacity_effect
        main_widget.preview_meta = preview_meta
        upload_btn.clicked.connect(lambda: self.upload_image(domain_name))
        diagnose_btn.clicked.connect(lambda: self.run_diagnosis(domain_name))
        cancel_btn.clicked.connect(lambda: self.cancel_diagnosis(domain_name))
        pdf_button.clicked.connect(lambda: self.save_report_as_pdf(domain_name))
        return main_widget

    # --- IMAGE PREVIEW & VALIDATION ---
    def set_image(self, file_path, domain):
        tab = self.domain_tabs[domain]
        is_valid, meta = self.validate_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Image", meta)
            tab.image_label.clear()
            tab.preview_meta.setText("")
            tab.result_display.clear()
            tab.reference_image_label.clear()
            return
        pixmap = QPixmap(file_path)
        tab.image_label.setPixmap(pixmap.scaled(
            tab.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        self.current_image_paths[domain] = file_path
        tab.symptom_input.clear()
        tab.result_display.clear()
        tab.pdf_button.setEnabled(False)
        tab.result_group.setVisible(False)
        tab.diagnosis_data = None
        tab.reference_image_label.setText("Reference image will appear here.")
        tab.reference_image_label.clear()
        tab.preview_meta.setText(meta)  # Show metadata for preview

    def upload_image(self, domain):
        image_folder = self.settings.value("image_folder", os.path.expanduser("~"))
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", image_folder,
                                                   "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_image(file_path, domain)

    def validate_image(self, file_path):
        # Only png, jpg, jpeg allowed
        valid_exts = ('.png', '.jpg', '.jpeg')
        if not file_path.lower().endswith(valid_exts):
            return False, "Supported image types: PNG, JPG, JPEG."
        try:
            size_bytes = os.path.getsize(file_path)
            max_size = 5 * 1024 * 1024  # 5MB
            if size_bytes > max_size:
                return False, f"Image is too large ({size_bytes // 1024} KB). Max allowed size: {max_size // 1024} KB."
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                return False, "Failed to load image file."
            width = pixmap.width()
            height = pixmap.height()
            meta = (f"Preview: {os.path.basename(file_path)}\n"
                    f"Dimensions: {width} x {height} px | Size: {size_bytes // 1024} KB | Format: {os.path.splitext(file_path)[-1][1:].upper()}")
            return True, meta
        except Exception as ex:
            return False, f"Error reading image: {ex}"

    # --- USER PREFERENCES / SETTINGS DIALOG ---
    def open_settings_dialog(self):
        before_theme = self.settings.value("theme", "Light")
        result = SettingsDialog.get_settings(self)
        if result:
            # Apply theme immediately
            self.apply_theme(result.get("theme", "Light"))
            # Optionally: update file dialogs' default folder on next use

    # --- DIAGNOSIS LOGIC ---
    def run_diagnosis(self, domain):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A diagnosis is already in progress. Please wait or cancel it.")
            return
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()
        if not image_path and not symptoms:
            QMessageBox.warning(self, "Input Missing", "Please upload an image OR describe the symptoms to proceed.")
            return
        if image_path and symptoms:
            tab.symptom_input.clear()
            symptoms = ""
        tab.diagnose_btn.setEnabled(False)
        tab.diagnose_btn.setVisible(False)
        tab.cancel_btn.setVisible(True)
        tab.pdf_button.setEnabled(False)
        tab.spinner.start()
        tab.result_display.setPlainText("Starting diagnosis...")
        tab.result_group.setVisible(False)
        self.worker_thread = QThread()
        self.diagnosis_worker = DiagnosisWorker(
            self.ml_processor, image_path, symptoms, domain, self.database
        )
        self.diagnosis_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.diagnosis_worker.run)
        self.diagnosis_worker.finished.connect(self.on_diagnosis_complete)
        self.diagnosis_worker.error.connect(self.on_diagnosis_error)
        self.diagnosis_worker.progress.connect(lambda msg: tab.result_display.setPlainText(msg))
        self.diagnosis_worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.diagnosis_worker.error.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def animate_result_fade_in(self, tab):
        tab.result_group.setVisible(True)
        animation = QPropertyAnimation(tab.result_opacity_effect, b"opacity")
        animation.setDuration(750)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        animation.start()
        self.result_animation = animation

    def cancel_diagnosis(self, domain):
        if self.worker_thread and self.worker_thread.isRunning():
            try:
                if self.diagnosis_worker:
                    self.diagnosis_worker.stop()
            except Exception:
                pass
            self.cleanup_worker_and_ui(domain, canceled=True)

    def on_diagnosis_complete(self, result, confidence, wiki_summary, predicted_stage, pubmed_summary, domain):
        self.cleanup_worker_and_ui(domain)
        tab = self.domain_tabs[domain]
        tab.diagnosis_data = {**result, 'confidence': confidence, 'stage': predicted_stage,
                              'image_path': self.current_image_paths[domain]}
        stages_str = "\n".join([f"  • <b>{k}:</b> {v}" for k, v in result.get("stages", {}).items()])
        output_html = (
            f"<h2 style='font-family: Arial, sans-serif; font-size: 18px; color: #2c3e50; margin-bottom: 10px;'>Diagnosis: {result.get('name', 'N/A')}</h2>"
            f"<p style='font-size: 14px;'><b>Confidence Score:</b> <span style='color:#27ae60; font-weight:bold; font-size: 16px;'>{confidence:.1f}%</span></p>"
            f"<p style='font-size: 14px;'><b>Predicted Stage:</b> <span style='color:#3498db;'>{predicted_stage}</span></p>"
            f"<p style='font-size: 14px;'><b>Description:</b><br><span style='color:#34495e;'>{result.get('description', 'N/A')}</span></p>"
            f"<p style='font-size: 14px;'><b>Wikipedia Summary:</b><br><span style='color:#34495e;'>{wiki_summary if wiki_summary else 'No summary available.'}</span></p>"
            f"<p style='font-size: 14px;'><b>Known Stages:</b><br><span style='color:#34495e;'>{stages_str if stages_str else 'No stages information available.'}</span></p>"
            f"<p style='font-size: 14px;'><b style='color:#e74c3c;'>Solution/Cure:</b><br><span style='color:#e74c3c; font-weight: bold;'>{result.get('solution', 'No solution available.')}</span></p>"
            f"<p style='font-size: 14px;'><b>Recent Research (PubMed):</b><br><span style='color:#34495e;'>{pubmed_summary if pubmed_summary else 'No recent research available.'}</span></p>"
        )
        tab.result_display.setHtml(output_html)
        tab.pdf_button.setEnabled(True)
        relative_image_path = result.get('image_url')
        if relative_image_path:
            full_image_path = os.path.join(self.base_app_dir, relative_image_path)
            if os.path.exists(full_image_path):
                pixmap = QPixmap(full_image_path)
                tab.reference_image_label.setPixmap(
                    pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
                tab.reference_image_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 10px; background-color: #ffffff; padding: 5px;")
            else:
                tab.reference_image_label.setText(f"Reference image not found:\n{relative_image_path}\nPlease check the database.")
                tab.reference_image_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 10px; background-color: #f8d7da; color: #721c24; padding: 5px;")
        else:
            tab.reference_image_label.setText("No reference image available in database.")
            tab.reference_image_label.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #ecf0f1; color: #7f8c8d; padding: 5px;")
        self.animate_result_fade_in(tab)
        location = tab.location_input.text().strip()
        if location:
            tab.location_input.clear()
            self.diagnosis_locations.append({"disease": result.get('name', 'N/A'), "location": location})
            self.status_bar.showMessage(f"Diagnosis complete. Location '{location}' logged.", 5000)
        else:
            self.status_bar.showMessage("Diagnosis complete.", 3000)

    def on_diagnosis_error(self, error_message, domain):
        self.cleanup_worker_and_ui(domain)
        tab = self.domain_tabs[domain]
        tab.result_display.setHtml(f"<h3 style='color:red;'>Diagnosis Failed</h3><p>{error_message}</p>")
        self.animate_result_fade_in(tab)
        self.status_bar.showMessage("Diagnosis failed.", 4000)

    def cleanup_worker_and_ui(self, domain, canceled=False):
        tab = self.domain_tabs[domain]
        tab.spinner.stop()
        tab.diagnose_btn.setEnabled(True)
        tab.diagnose_btn.setVisible(True)
        tab.cancel_btn.setVisible(False)
        if canceled:
            tab.result_display.setHtml("<p style='color:#dc3545;'>Diagnosis was canceled by the user.</p>")
            self.animate_result_fade_in(tab)
            self.status_bar.showMessage("Diagnosis canceled.", 4000)
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        self.worker_thread = None
        self.diagnosis_worker = None

    def open_add_disease_dialog(self):
        dialog = AddNewDiseaseDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not all([data.get('name'), data.get('description'), data.get('solution')]):
                QMessageBox.warning(self, "Incomplete Data",
                                    "Please fill in at least the Name, Description, and Solution fields.")
                return
            success, error_msg = save_disease(data)
            if success:
                self.database = load_database()
                QMessageBox.information(self, "Success",
                                        f"Disease '{data.get('name', '')}' has been saved successfully.")
            else:
                QMessageBox.critical(self, "Save Error", f"Failed to save the disease:\n{error_msg}")
        dialog.deleteLater()

    def save_report_as_pdf(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Data", "Please run a diagnosis before saving a report.")
            return
        pdf_folder = self.settings.value("pdf_folder", os.path.expanduser("~"))
        safe_name = re.sub(r'[\s/:*?"<>|]+', '_', tab.diagnosis_data.get('name', '')).lower()
        default_path = os.path.join(pdf_folder, f"{safe_name}_report.pdf")
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
        dialog.deleteLater()

    def open_image_search_dialog(self):
        dialog = ImageSearchDialog(self.database, self)
        dialog.exec()
        dialog.deleteLater()

    def open_map_dialog(self):
        dialog = MapDialog(self.diagnosis_locations, self)
        dialog.exec()
        dialog.deleteLater()

    def closeEvent(self, event):
        try:
            if self.worker_thread and self.worker_thread.isRunning():
                if self.diagnosis_worker:
                    self.diagnosis_worker.stop()
                self.worker_thread.quit()
                if not self.worker_thread.wait(1500):
                    print("Warning: Worker thread did not terminate gracefully.")
        except Exception:
            pass
        event.accept()

