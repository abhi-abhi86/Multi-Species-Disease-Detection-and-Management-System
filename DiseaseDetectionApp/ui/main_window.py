from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar, QMenu, QApplication
)
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import Qt, pyqtSignal
from ui.add_disease_dialog import AddNewDiseaseDialog
from ui.chatbot_dialog import ChatbotDialog
from ui.image_search_dialog import ImageSearchDialog
from core.data_handler import load_database, save_disease
# Updated import to use the new MLProcessor class
from core.ml_processor import MLProcessor, predict_from_symptoms

class DropLabel(QLabel):
    fileDropped = pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop an Image Here\nor Click 'Upload Image'")
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 8px; color: #555;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # Check if any of the files are valid images
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    ext = file_path.split('.')[-1].lower()
                    if ext in ['png', 'jpg', 'jpeg']:
                        event.acceptProposedAction()
                        self.setStyleSheet("border: 2px solid #68b2f8; background-color: #e8f4ff; border-radius: 8px;")
                        return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 8px; color: #555;")

    def dropEvent(self, event):
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0; border-radius: 8px; color: #555;")
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                url = urls[0] # Handle the first dropped file
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    ext = file_path.split('.')[-1].lower()
                    if ext in ['png', 'jpg', 'jpeg']:
                        self.fileDropped.emit(file_path)
                        event.acceptProposedAction()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Multi-Species Disease Management System")
        self.resize(800, 700)
        self.database = load_database()
        
        # Load the AI model processor when the application starts.
        # The first time you run this, it will download the model weights (~14MB).
        self.ml_processor = MLProcessor()

        self.current_image_paths = {"Plant": None, "Human": None, "Animal": None}
        self.setup_menu()
        self.tab_widget = QTabWidget()
        self.domain_tabs = {}
        for domain, label in zip(["Plant", "Human", "Animal"], ["🌱 Plants", "🧑 Humans", "🐾 Animals"]):
            tab = self.create_domain_tab(domain)
            self.tab_widget.addTab(tab, label)
            self.domain_tabs[domain] = tab
        self.setCentralWidget(self.tab_widget)

    def setup_menu(self):
        menubar = QMenuBar(self)
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

        # New "Tools" menu for image search
        tools_menu = QMenu("Tools", self)
        search_image_action = QAction("Search Disease Image...", self)
        search_image_action.triggered.connect(self.open_image_search_dialog)
        tools_menu.addAction(search_image_action)
        menubar.addMenu(tools_menu)

        self.setMenuBar(menubar)

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()

        image_label = DropLabel()
        image_label.setFixedSize(256, 256)
        image_label.fileDropped.connect(lambda path: self.set_image(path, domain_name))

        upload_btn = QPushButton("Upload Image")
        symptom_input = QTextEdit()
        symptom_input.setPlaceholderText("Or describe the symptoms here...")
        diagnose_btn = QPushButton("Diagnose")
        input_layout.addWidget(image_label, 0, 0, 2, 1)
        input_layout.addWidget(upload_btn, 2, 0)
        input_layout.addWidget(QLabel("Symptoms Description:"), 0, 1)
        input_layout.addWidget(symptom_input, 1, 1, 1, 2)
        input_layout.addWidget(diagnose_btn, 2, 1, 1, 2)
        input_group.setLayout(input_layout)
        result_group = QGroupBox("Diagnosis Results")
        result_layout = QVBoxLayout()
        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_layout.addWidget(result_display)
        result_group.setLayout(result_layout)
        main_layout.addWidget(input_group)
        main_layout.addWidget(result_group)
        main_widget.setLayout(main_layout)
        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
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

    def upload_image(self, domain):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_image(file_path, domain)

    def run_diagnosis(self, domain):
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()
        result, confidence, wiki_summary, predicted_stage = None, 0, "", ""

        use_symptoms = bool(symptoms)
        use_image = bool(image_path)

        # If both image and symptoms are provided, ask the user which to use
        if use_symptoms and use_image:
            reply = QMessageBox.question(self, 'Confirm Diagnosis Method',
                                         "Both an image and symptoms are provided. Do you want to diagnose based on symptoms? (Choosing 'No' will use the image).",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.No:
                use_symptoms = False # User wants to use the image
            else:
                use_image = False # User wants to use symptoms

        if use_symptoms:
            result, confidence, wiki_summary, predicted_stage = predict_from_symptoms(symptoms, domain, self.database)
        elif use_image:
            tab.result_display.setPlainText("Analyzing image... Please wait.")
            QApplication.processEvents()  # Update the UI to show the message
            result, confidence, wiki_summary, predicted_stage = self.ml_processor.predict_from_image(image_path, domain, self.database)
        else:
            QMessageBox.warning(self, "Input Missing", "Please upload an image or describe symptoms.")
            return

        if result:
            stages_str = "\n".join([f"  • {k}: {v}" for k, v in result.get("stages", {}).items()])
            out = (
                f"--- DIAGNOSIS ---\n"
                f"Confidence: {confidence:.1f}%\n"
                f"Disease Name: {result['name']}\n"
                f"Predicted Stage: {predicted_stage}\n\n"
                f"--- WIKIPEDIA SUMMARY ---\n{wiki_summary}\n\n"
                f"--- DETAILS FROM DATABASE ---\n"
                f"Description: {result['description']}\n\n"
                f"Known Stages:\n{stages_str if stages_str else '  • N/A'}\n\n"
                f"Common Causes:\n  • {result.get('causes', 'N/A')}\n\n"
                f"--- RECOMMENDATIONS ---\n"
                f"Solution/Cure:\n  • {result.get('solution', 'N/A')}"
            )
            tab.result_display.setPlainText(out)
        else:
            tab.result_display.setPlainText("No diagnosis could be made. The AI model could not identify a matching disease in the database for the provided input.")

    def open_add_disease_dialog(self):
        dialog = AddNewDiseaseDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            if not all([data['name'], data['description'], data['solution']]):
                QMessageBox.warning(self, "Incomplete Data", "Please fill in at least the Name, Description, and Solution.")
                return
            save_disease(data)
            self.database = load_database()
            QMessageBox.information(self, "Success", "Disease information saved.")
            
    def open_chatbot(self):
        dialog = ChatbotDialog(self.database, self)
        dialog.exec()

    def open_image_search_dialog(self):
        """Opens the image search dialog."""
        dialog = ImageSearchDialog(self.database, self)
        dialog.exec()
