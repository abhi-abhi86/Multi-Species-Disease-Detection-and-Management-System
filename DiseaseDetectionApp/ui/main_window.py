from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar, QMenu,
    QApplication, QLineEdit, QProgressBar, QStatusBar, QGraphicsOpacityEffect
)
from PyQt6.QtGui import QPixmap, QAction, QFont, QColor, QCursor
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve, QTimer
import os # Import the os module

from ui.add_disease_dialog import AddNewDiseaseDialog
from ui.chatbot_dialog import ChatbotDialog
from ui.image_search_dialog import ImageSearchDialog
from ui.map_dialog import MapDialog
from core.data_handler import load_database, save_disease
from core.ml_processor import MLProcessor
from core.worker import DiagnosisWorker

# --- Animated Button ---
class AnimatedButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_style = """
            QPushButton {
                background-color: #4f8cff;
                color: white;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 16px;
                transition: all 0.2s;
            }
            QPushButton:hover {
                background-color: #366fcc;
                box-shadow: 0 0 10px #4f8cff44;
            }
            QPushButton:pressed {
                background-color: #163b70;
            }
        """
        self.setStyleSheet(self.default_style)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

# --- Animated Drop Label ---
class AnimatedDropLabel(QLabel):
    fileDropped = pyqtSignal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("✨ Drag & Drop an Image Here\nor Click 'Upload Image'")
        self.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.setStyleSheet("""
            border: 2px dashed #4f8cff; background-color: #e8f0ff;
            border-radius: 12px; color: #366fcc;
            transition: box-shadow 0.2s;
        """)
        self.setMinimumHeight(180)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        border: 2px solid #366fcc; background-color: #d4e4ff;
                        border-radius: 12px; color: #163b70;
                        box-shadow: 0 0 20px #4f8cff77;
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            border: 2px dashed #4f8cff; background-color: #e8f0ff;
            border-radius: 12px; color: #366fcc;
        """)

    def dropEvent(self, event):
        self.setStyleSheet("""
            border: 2px dashed #4f8cff; background-color: #e8f0ff;
            border-radius: 12px; color: #366fcc;
        """)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.fileDropped.emit(url.toLocalFile())
                    event.acceptProposedAction()
                    return

# --- Animated Text Edit ---
class AnimatedTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QTextEdit {
                background: #f8faff;
                border-radius: 6px;
                font-size: 15px;
                border: 1px solid #e0e7ff;
                padding: 7px;
                transition: border 0.2s;
            }
            QTextEdit:focus {
                border: 1.5px solid #4f8cff;
                box-shadow: 0 0 8px #4f8cff22;
            }
        """)

# --- Fade Animation Helper ---
class FadeWidget(QWidget):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.effect = QGraphicsOpacityEffect()
        self.widget.setGraphicsEffect(self.effect)
        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(450)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def fade_in(self):
        self.anim.stop()
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()

    def fade_out(self):
        self.anim.stop()
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🦠 Multi-Species Disease Diagnosis (Animated)")
        self.resize(950, 760)
        self.setStyleSheet("background: #f5f7fb;")
        
        # --- NEW: Define base path for the application to find resources like images ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.app_base_dir = os.path.abspath(os.path.join(self.script_dir, '..'))
        
        self.database = load_database()
        self.ml_processor = MLProcessor()
        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.diagnosis_locations = []
        self.worker_thread = None
        self.diagnosis_worker = None
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabBar::tab:selected { background: #4f8cff; color: white; border-radius: 12px 12px 0 0;}
            QTabBar::tab { background: #e8f0ff; margin-right: 3px; padding: 11px 24px; font-size: 18px;}
        """)
        self.domain_tabs = {}
        for domain, label in zip(["Plant", "Human", "Animal"], ["🌱 Plants", "🧑 Humans", "🐾 Animals"]):
            tab = self.create_domain_tab(domain)
            self.tab_widget.addTab(tab, label)
            self.domain_tabs[domain] = tab
        self.setCentralWidget(self.tab_widget)
        self.setup_menu()

    def setup_menu(self):
        menubar = QMenuBar(self)
        menubar.setStyleSheet("QMenuBar {background: #e8f0ff; border-radius: 0 0 12px 12px;}")
        file_menu = QMenu("File", self)
        add_action = QAction("Add New Disease...", self)
        add_action.triggered.connect(self.open_add_disease_dialog)
        chatbot_action = QAction("Chatbot", self)
        chatbot_action.triggered.connect(self.open_chatbot)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(add_action)
        file_menu.addAction(chatbot_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)

        tools_menu = QMenu("Tools", self)
        search_image_action = QAction("Search Disease Image...", self)
        search_image_action.triggered.connect(self.open_image_search_dialog)
        map_action = QAction("View Disease Map", self)
        map_action.triggered.connect(self.open_map_dialog)
        tools_menu.addAction(search_image_action)
        tools_menu.addAction(map_action)
        menubar.addMenu(tools_menu)

        view_menu = QMenu("View", self)
        theme_action = QAction("Toggle Dark/Light Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        menubar.addMenu(view_menu)
        self.setMenuBar(menubar)

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(18, 18, 18, 18)

        input_group = QGroupBox("Input Data")
        input_group.setStyleSheet("""
            QGroupBox { font-size: 18px; border: 1.5px solid #4f8cff; border-radius: 12px; margin-top: 8px; background: #f0f6ff;}
            QGroupBox:title { top: -10px; left: 14px; padding: 0 8px;}
        """)
        input_layout = QGridLayout()
        image_label = AnimatedDropLabel()
        image_label.setFixedSize(240, 180)
        image_label.fileDropped.connect(lambda path: self.set_image(path, domain_name))
        upload_btn = AnimatedButton("Upload Image")
        upload_btn.setMinimumHeight(38)
        symptom_input = AnimatedTextEdit()
        symptom_input.setPlaceholderText("Or describe the symptoms here...")
        location_input = QLineEdit()
        location_input.setPlaceholderText("Optional: Enter location (e.g., City, Country)")
        location_input.setStyleSheet("""
            QLineEdit { background: #f8faff; border: 1px solid #e0e7ff; border-radius: 6px; font-size: 15px; padding: 7px;}
            QLineEdit:focus { border: 1.5px solid #4f8cff; box-shadow: 0 0 8px #4f8cff22;}
        """)

        diagnose_btn = AnimatedButton("Diagnose")
        diagnose_btn.setMinimumHeight(42)

        # Animated Progress Bar
        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setStyleSheet("""
            QProgressBar {border-radius: 8px; border: 1px solid #4f8cff; text-align: center; background-color: #e0e7ff; height: 24px; font-size: 15px;}
            QProgressBar::chunk {background-color: #4f8cff; border-radius: 8px;}
        """)

        input_layout.addWidget(image_label, 0, 0, 3, 1)
        input_layout.addWidget(upload_btn, 3, 0)
        input_layout.addWidget(QLabel("Symptoms Description:"), 0, 1)
        input_layout.addWidget(symptom_input, 1, 1, 1, 2)
        input_layout.addWidget(QLabel("Location:"), 2, 1)
        input_layout.addWidget(location_input, 2, 2)
        input_layout.addWidget(diagnose_btn, 3, 1, 1, 2)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        main_layout.addWidget(progress_bar)

        result_group = QGroupBox("Diagnosis Results")
        result_group.setStyleSheet("""
            QGroupBox { font-size: 18px; border: 1.5px solid #00b894; border-radius: 12px; margin-top: 8px; background: #f3fff8;}
            QGroupBox:title { top: -10px; left: 14px; padding: 0 8px;}
        """)

        # --- MODIFICATION: Use a QGridLayout for better result layout ---
        result_layout = QGridLayout()
        
        result_display = AnimatedTextEdit()
        result_display.setReadOnly(True)
        result_display.setStyleSheet(result_display.styleSheet() + """
            QTextEdit { background: #eafff1; color: #1c4034; font-weight: 500;}
        """)

        # --- NEW: Add a label for the reference image from the database ---
        reference_image_label = QLabel("Reference image will appear here.")
        reference_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reference_image_label.setWordWrap(True)
        reference_image_label.setFixedSize(220, 220)
        reference_image_label.setStyleSheet("""
            border: 1px solid #00b894;
            background-color: #eafff1;
            border-radius: 10px;
            color: #1c4034;
        """)
        
        result_fader = FadeWidget(result_display)
        
        # Add widgets to the new grid layout
        result_layout.addWidget(reference_image_label, 0, 0)
        result_layout.addWidget(result_display, 0, 1)
        result_layout.setColumnStretch(1, 1) # Make the text column expand

        result_group.setLayout(result_layout)

        main_layout.addWidget(result_group)
        main_widget.setLayout(main_layout)

        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
        # --- NEW: Add reference_image_label to the tab's widget dictionary ---
        main_widget.reference_image_label = reference_image_label
        main_widget.location_input = location_input
        main_widget.diagnose_btn = diagnose_btn
        main_widget.progress_bar = progress_bar
        main_widget.result_fader = result_fader

        upload_btn.clicked.connect(lambda: self.upload_image(domain_name))
        diagnose_btn.clicked.connect(lambda: self.run_diagnosis(domain_name))
        return main_widget

    def set_image(self, file_path, domain):
        if file_path:
            tab = self.domain_tabs[domain]
            pixmap = QPixmap(file_path)
            tab.image_label.setPixmap(pixmap.scaled(tab.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.current_image_paths[domain] = file_path
            tab.symptom_input.clear()
            tab.result_display.clear()
            tab.result_fader.fade_in()

    def upload_image(self, domain):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_image(file_path, domain)

    def run_diagnosis(self, domain):
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()
        tab.progress_bar.setVisible(True)
        tab.progress_bar.setMaximum(0)  # Indeterminate/Animated
        tab.result_fader.fade_out()

        if not image_path and not symptoms:
            QMessageBox.warning(self, "Input Missing", "Please upload an image or describe symptoms.")
            tab.progress_bar.setVisible(False)
            return

        tab.diagnose_btn.setEnabled(False)
        tab.result_display.setPlainText("Starting diagnosis...")

        use_symptoms = bool(symptoms)
        if use_symptoms and image_path:
            reply = QMessageBox.question(
                self, 'Confirm Diagnosis Method',
                "Both image and symptoms are provided. Diagnose with symptoms? (No uses the image).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.No:
                use_symptoms = False

        self.worker_thread = QThread()
        worker_image_path = None if use_symptoms else image_path
        worker_symptoms = symptoms if use_symptoms else ""

        self.diagnosis_worker = DiagnosisWorker(self.ml_processor, worker_image_path, worker_symptoms, domain, self.database)
        self.diagnosis_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.diagnosis_worker.run)
        
        self.diagnosis_worker.finished.connect(self.on_diagnosis_complete)
        self.diagnosis_worker.error.connect(self.on_diagnosis_error)
        
        self.diagnosis_worker.progress.connect(lambda msg: tab.result_display.setPlainText(msg))
        self.diagnosis_worker.finished.connect(self.stop_worker)
        self.diagnosis_worker.error.connect(self.stop_worker)
        self.worker_thread.start()

    def stop_worker(self):
        for tab in self.domain_tabs.values():
            tab.progress_bar.setVisible(False)
            tab.diagnose_btn.setEnabled(True)
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.diagnosis_worker = None

    def on_diagnosis_complete(self, result, confidence, wiki_summary, predicted_stage, pubmed_summary, domain):
        tab = self.domain_tabs[domain]
        stages_str = "\n".join([f"  • {k}: {v}" for k, v in result.get("stages", {}).items()])
        
        # --- FIX: Added PubMed summary to the final output display ---
        output_html = (
            f"<b>Confidence:</b> <span style='color:#00b894'>{confidence:.1f}%</span><br>"
            f"<b>Disease Name:</b> {result['name']}<br>"
            f"<b>Predicted Stage:</b> {predicted_stage}<br><br>"
            f"<b>Wikipedia Summary:</b><br>{wiki_summary}<br><br>"
            f"<b>Details:</b><br>{result.get('description', 'N/A')}<br><br>"
            f"<b>Known Stages:</b><br>{stages_str if stages_str else '  • N/A'}<br><br>"
            f"<b>Common Causes:</b> {result.get('causes', 'N/A')}<br>"
            f"<b>Risk Factors:</b> {result.get('risk_factors', 'N/A')}<br>"
            f"<b>Preventive Measures:</b> {result.get('preventive_measures', 'N/A')}<br><br>"
            f"<b><h3 style='color:#0984e3;'>Solution/Cure:</h3></b><p style='color:#0984e3;'>{result.get('solution', 'N/A')}</p><br>"
            f"<b><h3 style='color:#6c5ce7;'>Recent Research (PubMed):</h3></b><p>{pubmed_summary}</p>"
        )
        
        tab.result_display.setHtml(output_html)
        tab.result_fader.fade_in()

        # --- NEW FEATURE: Load and display the reference image from the database ---
        # Check for both 'image' and 'image_url' keys for compatibility
        relative_image_path = result.get('image') or result.get('image_url')
        
        if relative_image_path:
            # Construct the full path to the image relative to the application's base directory
            full_image_path = os.path.join(self.app_base_dir, relative_image_path)
            
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
            tab.reference_image_label.setText("No reference image in database.")
        # --- END NEW FEATURE ---

        location = tab.location_input.text().strip()
        if location and result:
            self.diagnosis_locations.append({
                "disease": result['name'],
                "location": location
            })
            self.status_bar.showMessage(f"Diagnosis complete. Location '{location}' logged.", 4000)
            tab.location_input.clear()
        else:
            self.status_bar.showMessage("Diagnosis complete", 2500)

    def on_diagnosis_error(self, error_message, domain):
        tab = self.domain_tabs[domain]
        tab.result_display.setPlainText(f"Diagnosis Failed:\n{error_message}")
        tab.result_fader.fade_in()
        self.status_bar.showMessage("Diagnosis failed", 3500)

    def open_add_disease_dialog(self):
        dialog = AddNewDiseaseDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            if not all([data['name'], data['description'], data['solution']]):
                QMessageBox.warning(self, "Incomplete Data", "Please fill in at least the Name, Description, and Solution.")
                return
            
            # --- FIX: Use the new save_disease function and handle its return value ---
            success, error_msg = save_disease(data)
            if success:
                self.database = load_database() # Reload the database to include the new entry
                QMessageBox.information(self, "Success", f"Disease '{data['name']}' saved successfully.")
            else:
                QMessageBox.critical(self, "Error", f"Failed to save disease: {error_msg}")


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
        if self.diagnosis_worker:
            self.diagnosis_worker.stop()
        self.stop_worker()
        event.accept()

    def toggle_theme(self):
        # Toggle between light/dark themes with animation
        if self.styleSheet() == "" or "background: #f5f7fb;" in self.styleSheet():
            self.setStyleSheet("""
                QMainWindow {background: #232629; color: #fff;}
                QTextEdit, QLineEdit {color: #fff; background: #353535;}
                QTabBar::tab:selected { background: #0984e3; color: white;}
                QTabBar::tab { background: #232629; color: #fff;}
                QMenuBar {background: #353535; color: #fff;}
                QGroupBox {background: #232629; color: #fff;}
            """)
        else:
            self.setStyleSheet("background: #f5f7fb;")

