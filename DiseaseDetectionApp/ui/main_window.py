

WATERMARK_AUTHOR = "abhi-abhi86"
WATERMARK_CHECK = True

def check_watermark():
    """Check if watermark is intact. Application will fail if removed."""
    if not WATERMARK_CHECK or WATERMARK_AUTHOR != "abhi-abhi86":
        print("ERROR: Watermark protection violated. Application cannot start.")
        print("Made by: abhi-abhi86")
        import sys
        sys.exit(1)


check_watermark()

import csv
import os
import re
import sys
import time
import traceback
import webbrowser
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar, QCheckBox,
    QLineEdit, QStatusBar, QDialog, QHBoxLayout, QGraphicsOpacityEffect,
    QFormLayout, QComboBox, QMenu
)
from PySide6.QtGui import QPixmap, QFont, QCursor, QMovie
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal, QThread, QPropertyAnimation, QEasingCurve, QTimer, QSettings, QEvent


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from ui.add_disease_dialog import AddNewDiseaseDialog
except ImportError:
    AddNewDiseaseDialog = None
try:
    from ui.chatbot_dialog import ChatbotDialog
except ImportError:
    ChatbotDialog = None
try:
    from ui.image_search_dialog import ImageSearchDialog
except ImportError:
    ImageSearchDialog = None
try:
    from ui.map_dialog import MapDialog
except ImportError:
    MapDialog = None
try:
    from core.data_handler import load_database, save_disease
except ImportError:
    load_database = None
    save_disease = None
try:
    from core.ml_processor import MLProcessor
except ImportError:
    MLProcessor = None
try:
    from core.worker import DiagnosisWorker
except ImportError:
    DiagnosisWorker = None
try:
    from core.report_generator import generate_pdf_report
except ImportError:
    generate_pdf_report = None
try:
    from core.html_report_generator import generate_html_report
except ImportError:
    generate_html_report = None
try:
    from core.llm_integrator import LLMIntegrator
except ImportError:
    LLMIntegrator = None
try:
    from core.search_engine import SearchEngine
except ImportError:
    SearchEngine = None

try:
    from core.update_worker import UpdateWorker
except ImportError:
    UpdateWorker = None


try:
    llm_available = True
except ImportError:
    llm_available = False
    LLMIntegrator = None


missing_deps = []
if load_database is None:
    missing_deps.append("data_handler")
if MLProcessor is None:
    missing_deps.append("ml_processor")
if DiagnosisWorker is None:
    missing_deps.append("worker")

if missing_deps:
    print(f"Warning: Missing critical dependencies: {', '.join(missing_deps)}")
    print("Application may not function properly. Please check the core modules.")


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

        self.image_folder = QLineEdit(self.settings.value("image_folder", os.path.expanduser("~")))
        self.pdf_folder = QLineEdit(self.settings.value("pdf_folder", os.path.expanduser("~")))

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(self.settings.value("theme", "Light"))
        self.auto_update_check_box = QCheckBox()
        self.auto_update_check_box.setChecked(self.settings.value("auto_update_check", True, type=bool))

        self.layout.addRow("Default Image Folder:", self.image_folder)
        self.layout.addRow("Default PDF Save Folder:", self.pdf_folder)
        self.layout.addRow("Theme:", self.theme_combo)
        self.layout.addRow("Check for updates on startup:", self.auto_update_check_box)
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
        self.settings.setValue("auto_update_check", self.auto_update_check_box.isChecked())
        super().accept()

    @staticmethod
    def get_settings(parent=None):
        dlg = SettingsDialog(parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = QSettings("DiseaseDetectionApp", "UserPreferences")
            return {
                "image_folder": s.value("image_folder", os.path.expanduser("~")),
                "pdf_folder": s.value("pdf_folder", os.path.expanduser("~")),
                "theme": s.value("theme", "Light"),
                "auto_update_check": s.value("auto_update_check", True, type=bool)
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
    fileDropped = Signal(str)

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
        try:
            self.ml_processor = MLProcessor()
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Model File Missing",
                                 f"A required model file could not be found:\n\n{e}\n\nThe diagnosis feature will be disabled. Please ensure all model files are correctly placed and restart the application.")
            self.ml_processor = None
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error",
                                 f"An unexpected error occurred during ML processor initialization:\n\n{e}\n\nDiagnosis features will be disabled.")
            self.ml_processor = None
        self.llm_integrator = LLMIntegrator() if llm_available else None
        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.diagnosis_locations = []
        self.diagnosis_worker = None
        self.base_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.result_animation = None
        self.diagnosis_start_time = None
        self.is_diagnosis_running = False
        self.worker_thread = QThread(self)
        self.worker_thread.start()

        self.setWindowOpacity(0.0)
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

        if self.settings.value("auto_update_check", True, type=bool):
            QTimer.singleShot(3000, lambda: self.check_for_updates(silent=True))

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        add_action = QAction("Add New Disease...", self)
        add_action.setShortcut("Ctrl+N")
        add_action.setStatusTip("Add a new disease to the database")
        add_action.triggered.connect(self.open_add_disease_dialog)
        file_menu.addAction(add_action)

        import_action = QAction("Import Disease Database...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setStatusTip("Import diseases from JSON file")
        import_action.triggered.connect(self.import_disease_database)
        file_menu.addAction(import_action)

        export_action = QAction("Export Disease Database...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Export current disease database")
        export_action.triggered.connect(self.export_disease_database)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu("Edit")
        clear_action = QAction("Clear All", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.setStatusTip("Clear all inputs and results")
        clear_action.triggered.connect(self.clear_all_inputs)
        edit_menu.addAction(clear_action)

        tools_menu = menubar.addMenu("Tools")
        chatbot_action = QAction("Disease Chatbot...", self)
        chatbot_action.setShortcut("Ctrl+B")
        chatbot_action.setStatusTip("Open AI-powered disease chatbot")
        chatbot_action.triggered.connect(self.open_chatbot)
        tools_menu.addAction(chatbot_action)

        search_image_action = QAction("Search Disease Image Online...", self)
        search_image_action.setShortcut("Ctrl+S")
        search_image_action.setStatusTip("Search for disease images online")
        search_image_action.triggered.connect(self.open_image_search_dialog)
        tools_menu.addAction(search_image_action)

        map_action = QAction("View Disease Map...", self)
        map_action.setShortcut("Ctrl+M")
        map_action.setStatusTip("View disease locations on map")
        map_action.triggered.connect(self.open_map_dialog)
        tools_menu.addAction(map_action)

        tools_menu.addSeparator()
        statistics_action = QAction("View Statistics...", self)
        statistics_action.setShortcut("Ctrl+T")
        statistics_action.setStatusTip("View diagnosis statistics")
        statistics_action.triggered.connect(self.show_statistics)
        tools_menu.addAction(statistics_action)

        retrain_action = QAction("Retrain Model...", self)
        retrain_action.setShortcut("Ctrl+R")
        retrain_action.setStatusTip("Retrain the ML model with current data")
        retrain_action.triggered.connect(self.manual_retrain_model)
        tools_menu.addAction(retrain_action)

        tools_menu.addSeparator()
        settings_action = QAction("Settings...", self)
        settings_action.setShortcut("Ctrl+P")
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self.open_settings_dialog)
        tools_menu.addAction(settings_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About...", self)
        about_action.setShortcut("F1")
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        developer_action = QAction("Developer Info...", self)
        developer_action.setShortcut("Ctrl+D")
        developer_action.setStatusTip("View developer information")
        developer_action.triggered.connect(self.show_developer_info)

        help_menu.addSeparator()
        update_action = QAction("Check for Updates...", self)
        update_action.setStatusTip("Check for a new version of the application")
        update_action.triggered.connect(lambda: self.check_for_updates(silent=False))
        help_menu.addAction(update_action)
        help_menu.addAction(developer_action)

    def setup_search_ui(self):
        """Sets up the global search bar and results list."""
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Instant Search for any disease...")
        self.search_bar.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 5px;")
        self.search_bar.setFixedWidth(300)
        
        # Add search bar to a toolbar for better placement
        toolbar = self.addToolBar("Search")
        toolbar.addWidget(QLabel("Search: "))
        toolbar.addWidget(self.search_bar)

        self.search_results_list = QListWidget(self)
        self.search_results_list.setWindowFlags(Qt.WindowType.Popup)
        self.search_results_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.search_results_list.setMouseTracking(True)
        self.search_results_list.itemClicked.connect(self.on_search_result_clicked)

        self.search_bar.textChanged.connect(self.on_search_query_changed)

    def on_search_query_changed(self, text):
        """Handles live updates as the user types in the search bar."""
        if not text.strip() or not self.search_engine:
            self.search_results_list.hide()
            return

        results = self.search_engine.search(text)
        self.search_results_list.clear()

        if results:
            for disease in results:
                item = QListWidgetItem(f"{disease.get('name')} ({disease.get('domain')})")
                item.setData(Qt.ItemDataRole.UserRole, disease) # Store the full object
                self.search_results_list.addItem(item)
            
            # Position and show the results list below the search bar
            pos = self.search_bar.mapToGlobal(self.search_bar.rect().bottomLeft())
            self.search_results_list.move(pos)
            self.search_results_list.setFixedWidth(self.search_bar.width())
            self.search_results_list.show()
        else:
            self.search_results_list.hide()

    def on_search_result_clicked(self, item):
        """Handles clicking a search result to display its details."""
        disease_data = item.data(Qt.ItemDataRole.UserRole)
        self.search_results_list.hide()
        self.search_bar.clear()
        
        # Find the correct tab and populate it with the data
        domain = disease_data.get('domain')
        if domain and domain in self.domain_tabs:
            # Switch to the correct domain tab
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i).lower() == domain.lower() + 's':
                    self.tab_widget.setCurrentIndex(i)
                    break
            
            # Simulate a diagnosis completion to display the data
            # We can set dummy values for fields not present in the search result
            self.on_diagnosis_complete(
                result=disease_data,
                confidence=100.0, # Indicate it's a direct lookup
                wiki_summary="Loaded from database via search.",
                predicted_stage="N/A",
                pubmed_summary="Please run a full diagnosis for recent research.",
                domain=domain
            )

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
        symptom_input.setStyleSheet("background-color: #ffffff; color: #2c3e50; border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px;")
        input_layout.addWidget(QLabel("Location:"), 2, 1)
        input_layout.addWidget(location_input, 2, 2)
        location_input.setStyleSheet("background-color: #ffffff; color: #2c3e50; border: 1px solid #bdc3c7; border-radius: 5px; padding: 5px;")
        input_layout.addWidget(button_container, 3, 1, 1, 2)
        input_layout.setRowStretch(6, 1)
        main_layout.addWidget(input_group)
        result_group = QGroupBox("2. Diagnosis Result")
        result_layout = QGridLayout(result_group)
        opacity_effect = QGraphicsOpacityEffect(result_group)
        result_group.setGraphicsEffect(opacity_effect)
        result_group.setVisible(False)
        main_widget.result_layout = result_layout
        reference_image_label = QLabel("Reference image will appear here.")
        reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_image_label.setFixedSize(300, 300)
        reference_image_label.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #ffffff; padding: 5px;")
        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_display.setStyleSheet("border: 2px solid #bdc3c7; border-radius: 10px; background-color: #ffffff; padding: 10px; font-family: Arial, sans-serif; font-size: 14px;")
        pdf_button = AnimatedButton("Save Report as PDF")
        pdf_button.setEnabled(False)
        html_button = AnimatedButton("Save Report as HTML")
        html_button.setEnabled(False)
        csv_button = AnimatedButton("Save Report as CSV")
        csv_button.setEnabled(False)
        chatbot_button = AnimatedButton("AI Chatbot Query")
        chatbot_button.setEnabled(False)
        image_search_button = AnimatedButton("Search Images Online")
        image_search_button.setEnabled(False)
        result_layout.addWidget(reference_image_label, 0, 0)
        result_layout.addWidget(result_display, 0, 1)

        reports_menu_button = AnimatedButton("Reports")
        reports_menu = QMenu(reports_menu_button)
        reports_menu.addAction("Save as PDF", lambda: self.save_report_as_pdf(domain_name))
        reports_menu.addAction("Save as HTML", lambda: self.save_report_as_html(domain_name))
        reports_menu.addAction("Save as CSV", lambda: self.save_report_as_csv(domain_name))
        reports_menu.addSeparator()
        reports_menu.addAction("AI Chatbot Query", lambda: self.open_chatbot_with_query(domain_name))
        reports_menu.addAction("Search Images Online", lambda: self.open_image_search_with_disease(domain_name))
        reports_menu_button.setMenu(reports_menu)
        reports_menu_button.setEnabled(False)
        result_layout.addWidget(reports_menu_button, 1, 0)
        result_layout.setColumnStretch(1, 1)
        main_layout.addWidget(result_group)
        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
        main_widget.reference_image_label = reference_image_label
        main_widget.location_input = location_input
        main_widget.diagnose_btn = diagnose_btn
        main_widget.cancel_btn = cancel_btn
        main_widget.reports_menu_button = reports_menu_button
        main_widget.spinner = spinner
        main_widget.diagnosis_data = None
        main_widget.result_group = result_group
        main_widget.result_opacity_effect = opacity_effect
        main_widget.preview_meta = preview_meta
        upload_btn.clicked.connect(lambda: self.upload_image(domain_name))
        diagnose_btn.clicked.connect(lambda: self.run_diagnosis(domain_name))
        cancel_btn.clicked.connect(lambda: self.cancel_diagnosis(domain_name))
        return main_widget


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
        tab.reports_menu_button.setEnabled(False)
        tab.result_group.setVisible(False)
        tab.diagnosis_data = None
        tab.reference_image_label.setText("Reference image will appear here.")
        tab.reference_image_label.clear()
        tab.preview_meta.setText(meta)

    def upload_image(self, domain):
        image_folder = self.settings.value("image_folder", os.path.expanduser("~"))
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", image_folder,
                                                   "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_image(file_path, domain)

    def validate_image(self, file_path):

        valid_exts = ('.png', '.jpg', '.jpeg')
        if not file_path.lower().endswith(valid_exts):
            return False, "Supported image types: PNG, JPG, JPEG."
        try:
            size_bytes = os.path.getsize(file_path)
            max_size = 5 * 1024 * 1024
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


    def open_settings_dialog(self):
        before_theme = self.settings.value("theme", "Light")
        result = SettingsDialog.get_settings(self)
        if result:

            self.apply_theme(result.get("theme", "Light"))



    def run_diagnosis(self, domain):
        if not self.ml_processor:
            QMessageBox.critical(self, "Diagnosis Disabled",
                                 "The diagnosis feature is currently disabled due to a missing model file or an initialization error. Please check the application setup.")
            return

        if self.is_diagnosis_running:
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
        tab.reports_menu_button.setEnabled(False)
        tab.spinner.start()
        tab.result_display.setPlainText("Starting diagnosis...")
        tab.result_group.setVisible(False)
        self.diagnosis_start_time = time.time()
        self.diagnosis_worker = DiagnosisWorker(
            self.ml_processor, image_path, symptoms, domain, self.database
        )
        # Use QueuedConnection to ensure UI updates happen on the main thread
        self.diagnosis_worker.finished.connect(self.on_diagnosis_complete, Qt.ConnectionType.QueuedConnection)
        self.diagnosis_worker.error.connect(self.on_diagnosis_error, Qt.ConnectionType.QueuedConnection)
        self.diagnosis_worker.progress.connect(self.update_progress_text, Qt.ConnectionType.QueuedConnection)

        self.diagnosis_worker.moveToThread(self.worker_thread)
        QTimer.singleShot(0, self.diagnosis_worker.run)
        self.is_diagnosis_running = True
    
    def update_progress_text(self, domain, message):
        """Slot to update progress text safely on the main thread"""
        tab = self.domain_tabs.get(domain)
        if tab:
            tab.result_display.setPlainText(message)
        
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
        if self.is_diagnosis_running:
            try:
                if self.diagnosis_worker:
                    self.diagnosis_worker.stop()
            except Exception:
                pass
            self.cleanup_worker_and_ui(domain, canceled=True)

    def on_diagnosis_complete(self, result, confidence, wiki_summary, predicted_stage, pubmed_summary, domain):
        self.cleanup_worker_and_ui(domain)
        tab = self.domain_tabs[domain]
        diagnosis_time = time.time() - self.diagnosis_start_time if self.diagnosis_start_time else 0
        tab.diagnosis_data = {**result, 
                              'confidence': confidence, 
                              'stage': predicted_stage,
                              'image_path': self.current_image_paths[domain], 
                              'diagnosis_time': diagnosis_time,
                              'wiki_summary': wiki_summary,
                              'pubmed_summary': pubmed_summary}


        if result.get('name') == "No Confident Match Found" or confidence < 50.0:

            add_disease_button = AnimatedButton("Add New Disease", primary_color="#17a2b8", hover_color="#138496")
            add_disease_button.clicked.connect(lambda: self.open_add_disease_dialog_with_prefill(domain))

            tab.result_layout.addWidget(add_disease_button, 2, 0)

        stages_str = "\n".join([f"  • <b>{k}:</b> {v}" for k, v in result.get("stages", {}).items()])
        causes = result.get('causes', 'No causes information available.')
        risk_factors = result.get('risk_factors', 'No risk factors information available.')
        preventive_measures = result.get('preventive_measures', [])
        if isinstance(preventive_measures, list):
            preventive_str = "\n".join([f"  • {measure}" for measure in preventive_measures])
        else:
            preventive_str = preventive_measures
        report_made_by = result.get('report_made_by', 'N/A')
        diagnosis_time_str = f"{diagnosis_time:.2f} seconds" if diagnosis_time > 0 else "N/A"
        output_html = (
            f"<h2 style='font-family: Arial, sans-serif; font-size: 18px; color: #2c3e50;'>{result.get('name', 'Unknown Disease')}</h2>"
            f"<p style='font-size: 14px;'><b>Confidence Score:</b> <span style='color: {'green' if confidence >= 70 else 'orange' if confidence >= 50 else 'red'};'>{confidence:.1f}%</span></p>"
            f"<p style='font-size: 14px;'><b>Predicted Stage:</b> <span style='color: #2c3e50;'>{predicted_stage}</span></p>"
            f"<p style='font-size: 14px;'><b>Diagnosis Time:</b> <span style='color: #2c3e50;'>{diagnosis_time_str}</span></p>"
            f"<p style='font-size: 14px;'><b>Description:</b><br><span style='color: #2c3e50;'>{result.get('description', 'No description available.')}</span></p>"
            f"<p style='font-size: 14px;'><b>Causes:</b><br><span style='color: #2c3e50;'>{causes}</span></p>"
            f"<p style='font-size: 14px;'><b>Risk Factors:</b><br><span style='color: #2c3e50;'>{risk_factors}</span></p>"
            f"<p style='font-size: 14px;'><b>Preventive Measures:</b><br><span style='color: #2c3e50;'>{preventive_str}</span></p>"
            f"<h3 style='font-family: Arial, sans-serif; font-size: 16px; color: #2c3e50; margin-top: 15px;'>Wikipedia Summary</h3>"
            f"<p style='font-size: 14px; color: #2c3e50; margin-bottom: 15px;'>{wiki_summary or 'No summary available.'}</p>"
            f"<p style='font-size: 14px;'><b>Known Stages:</b><br><span style='color: #2c3e50;'>{stages_str}</span></p>"
            f"<p style='font-size: 14px;'><b>Solution/Cure:</b><br><span style='color: #2c3e50;'>{result.get('solution', 'No solution available.')}</span></p>"
            f"<p style='font-size: 14px;'><b>Recent Research (PubMed):</b><br><span style='color: #2c3e50;'>{pubmed_summary or 'No recent research available.'}</span></p>"
            f"<p style='font-size: 14px;'><b>Report Made By:</b> <span style='color: #2c3e50;'>{report_made_by}</span></p>"
        )
        tab.result_display.setHtml(output_html)
        tab.reports_menu_button.setEnabled(True)
        relative_image_path = result.get('image_url')
        image_displayed = False
        if relative_image_path:
            full_image_path = os.path.join(self.base_app_dir, relative_image_path)
            if os.path.exists(full_image_path):
                pixmap = QPixmap(full_image_path)
                tab.reference_image_label.setPixmap(
                    pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
                tab.reference_image_label.setStyleSheet("border: 2px solid #27ae60; border-radius: 10px; background-color: #ffffff; padding: 5px;")
                image_displayed = True
            else:

                image_dir = os.path.dirname(full_image_path)
                if os.path.exists(image_dir):
                    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:

                        fallback_image_path = os.path.join(image_dir, image_files[0])
                        pixmap = QPixmap(fallback_image_path)
                        tab.reference_image_label.setPixmap(
                            pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
                        tab.reference_image_label.setStyleSheet("border: 2px solid #f39c12; border-radius: 10px; background-color: #fff3cd; padding: 5px;")

                        result['image_url'] = os.path.relpath(fallback_image_path, self.base_app_dir).replace('\\', '/')
                        image_displayed = True
                    else:

                        uploaded_image = self.current_image_paths.get(domain)
                        if uploaded_image and os.path.exists(uploaded_image):
                            pixmap = QPixmap(uploaded_image)
                            tab.reference_image_label.setPixmap(
                                pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
                            tab.reference_image_label.setStyleSheet("border: 2px solid #3498db; border-radius: 10px; background-color: #d4e6f1; padding: 5px;")

                            result['image_url'] = os.path.relpath(uploaded_image, self.base_app_dir).replace('\\', '/')
                            image_displayed = True
                        else:
                            tab.reference_image_label.setText(f"No reference images found in:\n{image_dir}\nPlease check the database.")
                            tab.reference_image_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 10px; background-color: #f8d7da; color: #721c24; padding: 5px;")
                else:

                    uploaded_image = self.current_image_paths.get(domain)
                    if uploaded_image and os.path.exists(uploaded_image):
                        pixmap = QPixmap(uploaded_image)
                        tab.reference_image_label.setPixmap(
                            pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
                        tab.reference_image_label.setStyleSheet("border: 2px solid #3498db; border-radius: 10px; background-color: #d4e6f1; padding: 5px;")

                        result['image_url'] = os.path.relpath(uploaded_image, self.base_app_dir).replace('\\', '/')
                        image_displayed = True
                    else:
                        tab.reference_image_label.setText(f"Reference image not found:\n{relative_image_path}\nPlease check the database.")
                        tab.reference_image_label.setStyleSheet("border: 2px solid #e74c3c; border-radius: 10px; background-color: #f8d7da; color: #721c24; padding: 5px;")
        else:

            uploaded_image = self.current_image_paths.get(domain)
            if uploaded_image and os.path.exists(uploaded_image):
                pixmap = QPixmap(uploaded_image)
                tab.reference_image_label.setPixmap(
                    pixmap.scaled(tab.reference_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
                tab.reference_image_label.setStyleSheet("border: 2px solid #3498db; border-radius: 10px; background-color: #d4e6f1; padding: 5px;")

                result['image_url'] = os.path.relpath(uploaded_image, self.base_app_dir).replace('\\', '/')
                image_displayed = True
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
            tab.result_display.setHtml("<p style='color: orange;'>Diagnosis canceled.</p>")
            self.animate_result_fade_in(tab)
            self.status_bar.showMessage("Diagnosis canceled.", 4000)
        self.diagnosis_worker = None
        self.is_diagnosis_running = False

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

    def open_add_disease_dialog_with_prefill(self, domain):
        """Open add disease dialog with current image path and domain pre-filled."""
        image_path = self.current_image_paths.get(domain)
        dialog = AddNewDiseaseDialog(self, image_path=image_path, domain=domain)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not all([data.get('name'), data.get('description'), data.get('solution')]):
                QMessageBox.warning(self, "Incomplete Data",
                                    "Please fill in at least the Name, Description, and Solution fields.")
                return
            success, error_msg = save_disease(data)
            if success:
                self.database = load_database()

                reply = QMessageBox.question(self, "Retrain Model",
                                           f"Disease '{data.get('name', '')}' has been saved successfully.\n\n"
                                           "Would you like to retrain the ML model now to include this new disease?\n"
                                           "This will take a few minutes but will allow immediate diagnosis of the new disease.",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

                if reply == QMessageBox.StandardButton.Yes:
                    self.start_background_retraining()
                    QMessageBox.information(self, "Retraining Started",
                                          "Model retraining has started in the background.\n"
                                          "You will be notified when it completes.")
                else:
                    QMessageBox.information(self, "Disease Added",
                                          f"Disease '{data.get('name', '')}' has been saved successfully.\n\n"
                                          "Note: The ML model has not been retrained yet. You can retrain it later using:\n"
                                          "Tools → Retrain Model...")

            else:
                QMessageBox.critical(self, "Save Error", f"Failed to save the disease:\n{error_msg}")
        dialog.deleteLater()

    def start_background_retraining(self):
        """Start model retraining in a background thread."""
        if hasattr(self, 'retraining_thread') and self.retraining_thread.isRunning():
            QMessageBox.warning(self, "Retraining in Progress", "Model retraining is already in progress.")
            return


        self.status_bar.showMessage("Retraining model in background, please wait...", 0)


        from core.retraining_worker import RetrainingWorker
        self.retraining_worker = RetrainingWorker()
        self.retraining_thread = QThread()
        self.retraining_worker.moveToThread(self.retraining_thread)

        self.retraining_thread.started.connect(self.retraining_worker.run)
        self.retraining_worker.finished.connect(self.on_retraining_complete)
        self.retraining_worker.error.connect(self.on_retraining_error)
        self.retraining_worker.finished.connect(self.retraining_thread.quit)
        self.retraining_thread.finished.connect(self.retraining_thread.deleteLater)

        self.retraining_thread.start()

    def on_retraining_complete(self):
        """Handle successful retraining completion."""
        self.status_bar.showMessage("Model retraining completed successfully!", 5000)

        self.database = load_database()
        self.ml_processor = MLProcessor()


        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user_added_dir = os.path.join(base_dir, 'user_added_diseases')

        new_diseases_count = 0
        if os.path.exists(user_added_dir):
            for root, dirs, files in os.walk(user_added_dir):
                for file in files:
                    if file.endswith('.json'):
                        new_diseases_count += 1

        message = "Model has been updated with all current disease data."
        if new_diseases_count > 0:
            message += f"\n\nSuccessfully trained {new_diseases_count} new disease(s)!"
        message += "\n\nYou can now diagnose all diseases in the database, including any newly added ones."

        QMessageBox.information(self, "Retraining Complete", message)

    def on_retraining_error(self, error_message):
        """Handle retraining error."""
        self.status_bar.showMessage("Model retraining failed.", 5000)
        QMessageBox.critical(self, "Retraining Error", f"Failed to retrain the model:\n{error_message}")

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
            success, error_msg = generate_pdf_report(tab.diagnosis_data, file_path)
            if success:
                QMessageBox.information(self, "Success", f"Report saved successfully to:\n{file_path}")
            else:
                QMessageBox.critical(self, "PDF Error", f"Failed to generate PDF report:\n{error_msg}")

    def save_report_as_html(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Data", "Please run a diagnosis before saving a report.")
            return
        html_folder = self.settings.value("pdf_folder", os.path.expanduser("~"))
        safe_name = re.sub(r'[\s/:*?"<>|]+', '_', tab.diagnosis_data.get('name', '')).lower()
        default_path = os.path.join(html_folder, f"{safe_name}_report.html")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HTML Report", default_path, "HTML Files (*.html)")
        if file_path:
            success, error_msg = generate_html_report(tab.diagnosis_data, file_path)
            if success:
                QMessageBox.information(self, "Success", f"Report saved successfully to:\n{file_path}")

                webbrowser.open(f"file://{file_path}")
            else:
                QMessageBox.critical(self, "HTML Error", f"Failed to generate HTML report:\n{error_msg}")

    def save_report_as_csv(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Data", "Please run a diagnosis before saving a report.")
            return
        csv_folder = self.settings.value("pdf_folder", os.path.expanduser("~"))
        safe_name = re.sub(r'[\s/:*?"<>|]+', '_', tab.diagnosis_data.get('name', '')).lower()
        default_path = os.path.join(csv_folder, f"{safe_name}_report.csv")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV Report", default_path, "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    data = tab.diagnosis_data
                    
                    # Define headers for a more standard CSV layout
                    fieldnames = [
                        'Disease Name', 'Confidence Score', 'Predicted Stage', 'Domain',
                        'Description', 'Causes', 'Risk Factors', 'Preventive Measures',
                        'Known Stages', 'Solution/Cure', 'Wikipedia Summary', 
                        'Recent Research (PubMed)', 'Diagnosis Time (s)', 'Input Image Path'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    # Prepare complex fields for single-cell storage
                    stages = data.get('stages', {})
                    stages_str = '; '.join([f"{k}: {v}" for k, v in stages.items()]) if stages else 'N/A'
                    
                    preventive = data.get('preventive_measures', [])
                    preventive_str = '; '.join(preventive) if isinstance(preventive, list) else preventive

                    # Clean summaries by removing newlines to keep the CSV to one row
                    wiki_summary = (data.get('wiki_summary') or '').replace('\n', ' ').replace('\r', '')
                    pubmed_summary = (data.get('pubmed_summary') or '').replace('\n', ' ').replace('\r', '')

                    # Write the single data row
                    writer.writerow({
                        'Disease Name': data.get('name', 'N/A'),
                        'Confidence Score': f"{data.get('confidence', 0):.1f}%",
                        'Predicted Stage': data.get('stage', 'N/A'),
                        'Domain': data.get('domain', 'N/A'),
                        'Description': data.get('description', 'N/A'),
                        'Causes': data.get('causes', 'N/A'),
                        'Risk Factors': data.get('risk_factors', 'N/A'),
                        'Preventive Measures': preventive_str,
                        'Known Stages': stages_str,
                        'Solution/Cure': data.get('solution', 'N/A'),
                        'Wikipedia Summary': wiki_summary,
                        'Recent Research (PubMed)': pubmed_summary,
                        'Diagnosis Time (s)': f"{data.get('diagnosis_time', 0):.2f}",
                        'Input Image Path': data.get('image_path', 'N/A')
                    })

                QMessageBox.information(self, "Success", f"CSV report saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "CSV Error", f"Failed to generate CSV report:\n{str(e)}")

    def open_chatbot(self):
        dialog = ChatbotDialog(self.database, self, llm_integrator=self.llm_integrator)
        dialog.exec()
        dialog.deleteLater()

    def open_chatbot_with_query(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Diagnosis", "Please run a diagnosis first to query the chatbot.")
            return
        disease_name = tab.diagnosis_data.get('name', '')
        dialog = ChatbotDialog(self.database, self, initial_query=disease_name, llm_integrator=self.llm_integrator)
        dialog.exec()
        dialog.deleteLater()

    def open_image_search_dialog(self):
        dialog = ImageSearchDialog(self.database, self)
        dialog.exec()
        dialog.deleteLater()

    def open_image_search_with_disease(self, domain):
        tab = self.domain_tabs[domain]
        if not tab.diagnosis_data:
            QMessageBox.warning(self, "No Diagnosis", "Please run a diagnosis first to search for images.")
            return
        disease_name = tab.diagnosis_data.get('name', '')

        query = f"{disease_name} disease images"
        url = f"https://www.google.com/search?tbm=isch&q={query.replace(' ', '+')}"
        webbrowser.open(url)

    def open_map_dialog(self):
        dialog = MapDialog(self.diagnosis_locations, self)
        dialog.exec()
        dialog.deleteLater()

    def showEvent(self, event):
        """Override showEvent to trigger fade-in animation when window is shown."""
        super().showEvent(event)

        self.fade_in_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in_animation.setDuration(1000)
        self.fade_in_animation.setStartValue(0.0)
        self.fade_in_animation.setEndValue(1.0)
        self.fade_in_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_in_animation.start()

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()



    def import_disease_database(self):
        """Import diseases from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Disease Database", "",
                                                   "JSON Files (*.json)")
        if file_path:
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    imported_data = json.load(f)


                if not isinstance(imported_data, list):
                    QMessageBox.warning(self, "Invalid Format",
                                        "The imported file must contain a list of disease objects.")
                    return


                imported_count = 0
                for disease_data in imported_data:
                    if isinstance(disease_data, dict) and 'name' in disease_data:
                        success, error_msg = save_disease(disease_data)
                        if success:
                            imported_count += 1
                        else:
                            print(f"Failed to import disease '{disease_data.get('name', 'Unknown')}': {error_msg}")

                if imported_count > 0:
                    self.database = load_database()
                    QMessageBox.information(self, "Import Successful",
                                            f"Successfully imported {imported_count} diseases.")
                else:
                    QMessageBox.warning(self, "Import Failed",
                                        "No diseases were successfully imported.")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import database:\n{str(e)}")

    def export_disease_database(self):
        """Export current disease database to JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Disease Database", "disease_database.json",
                                                   "JSON Files (*.json)")
        if file_path:
            try:
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.database, f, indent=2, ensure_ascii=False)

                QMessageBox.information(self, "Export Successful",
                                        f"Database exported successfully to:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export database:\n{str(e)}")

    def clear_all_inputs(self):
        """Clear all inputs and results across all tabs."""
        for domain in self.domain_tabs:
            tab = self.domain_tabs[domain]
            tab.image_label.clear()
            tab.image_label.setText("Drag & Drop an Image Here\nor Click 'Upload Image'")
            tab.symptom_input.clear()
            tab.location_input.clear()
            tab.result_display.clear()
            tab.reference_image_label.clear()
            tab.reference_image_label.setText("Reference image will appear here.")
            tab.reports_menu_button.setEnabled(False)
            tab.result_group.setVisible(False)
            tab.diagnosis_data = None
            tab.preview_meta.setText("")

        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.status_bar.showMessage("All inputs cleared.", 3000)

    def show_statistics(self):
        """Show diagnosis statistics dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit

        dialog = QDialog(self)
        dialog.setWindowTitle("Diagnosis Statistics")
        dialog.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout(dialog)


        total_diagnoses = len(self.diagnosis_locations)
        disease_counts = {}
        location_counts = {}

        for entry in self.diagnosis_locations:
            disease = entry['disease']
            location = entry['location']

            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            location_counts[location] = location_counts.get(location, 0) + 1


        stats_text = QTextEdit()
        stats_text.setReadOnly(True)

        stats_html = f"""
        <h2>Diagnosis Statistics</h2>
        <p><b>Total Diagnoses:</b> {total_diagnoses}</p>

        <h3>Diseases Diagnosed:</h3>
        <ul>
        {"".join(f"<li>{disease}: {count} times</li>" for disease, count in disease_counts.items())}
        </ul>

        <h3>Locations:</h3>
        <ul>
        {"".join(f"<li>{location}: {count} diagnoses</li>" for location, count in location_counts.items())}
        </ul>

        <h3>Database Summary:</h3>
        <p><b>Total Diseases in Database:</b> {len(self.database)}</p>
        """

        stats_text.setHtml(stats_html)
        layout.addWidget(stats_text)

        dialog.exec()

    def manual_retrain_model(self):
        """Manually trigger model retraining."""

        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user_added_dir = os.path.join(base_dir, 'user_added_diseases')

        new_diseases_count = 0
        if os.path.exists(user_added_dir):
            for root, dirs, files in os.walk(user_added_dir):
                for file in files:
                    if file.endswith('.json'):
                        new_diseases_count += 1

        message = "This will retrain the ML model with all current disease data"
        if new_diseases_count > 0:
            message += f", including {new_diseases_count} newly added disease(s)"
        message += ".\nThe process may take several minutes.\n\nContinue?"

        reply = QMessageBox.question(self, "Retrain Model", message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.start_background_retraining()

    def show_about_dialog(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("About")
        dialog.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout(dialog)

        about_text = QLabel()
        about_text.setText("""
        <h2>Multi-Species Disease Detection and Management System</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Description:</b> AI-powered disease detection system for plants, humans, and animals</p>
        <p><b>Technologies:</b> Python, PySide6, PyTorch, Machine Learning</p>
        <p><b>Developer:</b> Abhishek MG</p>
        """)
        about_text.setWordWrap(True)
        layout.addWidget(about_text)

        dialog.exec()

    def show_developer_info(self):
        """Show developer information dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel

        dialog = QDialog(self)
        dialog.setWindowTitle("Developer Information")
        dialog.setGeometry(300, 300, 500, 300)

        layout = QVBoxLayout(dialog)

        try:
            import json
            with open('.developer_info.json', 'r') as f:
                dev_info = json.load(f)

            info_text = QLabel()
            info_html = f"""
            <h2>Developer Information</h2>
            <p><b>Name:</b> {dev_info['developer']['name']}</p>
            <p><b>Username:</b> {dev_info['developer']['username']}</p>
            <p><b>Email:</b> {dev_info['developer']['email']}</p>
            <p><b>GitHub:</b> <a href="{dev_info['developer']['github']}">{dev_info['developer']['github']}</a></p>
            <p><b>Project:</b> {dev_info['developer']['project']}</p>
            <p><b>Description:</b> {dev_info['developer']['description']}</p>
            <p><b>Version:</b> {dev_info['developer']['version']}</p>
            <p><b>Last Updated:</b> {dev_info['developer']['last_updated']}</p>
            <h3>Features:</h3>
            <ul>
            {"".join(f"<li>{feature}</li>" for feature in dev_info['developer']['features'])}
            </ul>
            """
            info_text.setText(info_html)
            info_text.setWordWrap(True)
            info_text.setOpenExternalLinks(True)
            layout.addWidget(info_text)

        except Exception as e:
            error_label = QLabel(f"Error loading developer information: {str(e)}")
            layout.addWidget(error_label)

        dialog.exec()

    def check_for_updates(self, silent=False):
        if not UpdateWorker:
            if not silent:
                QMessageBox.critical(self, "Error", "The UpdateWorker component is missing.")
            return

        if hasattr(self, 'update_thread') and self.update_thread.isRunning():
            if not silent:
                QMessageBox.information(self, "In Progress", "An update check is already in progress.")
            return

        self._is_silent_update_check = silent

        if not silent:
            self.status_bar.showMessage("Checking for updates...", 2000)

        self.update_worker = UpdateWorker()
        self.update_thread = QThread()
        self.update_worker.moveToThread(self.update_thread)

        self.update_thread.started.connect(self.update_worker.run)
        self.update_worker.finished.connect(self.on_update_check_complete)
        self.update_worker.finished.connect(self.update_thread.quit)
        self.update_thread.finished.connect(self.update_thread.deleteLater)

        self.update_thread.start()

    def on_update_check_complete(self, update_info, error_message):
        if error_message:
            QMessageBox.critical(self, "Update Check Failed", f"An error occurred while checking for updates:\n\n{error_message}")
        elif update_info:
            latest_version = update_info.get("latest_version", "N/A")
            release_notes = update_info.get("release_notes", "No release notes provided.")
            release_url = update_info.get("release_url")

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("Update Available")
            msg_box.setText(f"A new version ({latest_version}) is available!")
            formatted_notes = release_notes.replace('\n', '<br>')
            msg_box.setInformativeText(f"<b>Release Notes:</b><br>{formatted_notes}")
            if release_url:
                msg_box.addButton("Download", QMessageBox.ButtonRole.AcceptRole).clicked.connect(lambda: webbrowser.open(release_url))
            msg_box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()
        elif not self._is_silent_update_check:
            QMessageBox.information(self, "Up to Date", "You are currently running the latest version of the application.")

        if not self._is_silent_update_check:
            self.status_bar.clearMessage()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
