from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar, QMenu
)
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import Qt
from ui.add_disease_dialog import AddNewDiseaseDialog
from ui.chatbot_dialog import ChatbotDialog
from core.data_handler import load_database, save_disease
from core.ml_processor import predict_from_image, predict_from_symptoms

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Multi-Species Disease Management System")
        self.resize(800, 700)
        self.database = load_database()
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
        self.setMenuBar(menubar)

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()
        image_label = QLabel("Upload an Image")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setFixedSize(256, 256)
        image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        upload_btn = QPushButton("Scan Disease (Upload Image)")
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

    def upload_image(self, domain):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            tab = self.domain_tabs[domain]
            pixmap = QPixmap(file_path)
            tab.image_label.setPixmap(pixmap.scaled(tab.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.current_image_paths[domain] = file_path
            tab.symptom_input.clear()

    def run_diagnosis(self, domain):
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()
        result, confidence, wiki_summary = None, 0, ""

        if image_path:
            result, confidence, wiki_summary = predict_from_image(image_path, domain, self.database)
        elif symptoms:
            result, confidence, wiki_summary = predict_from_symptoms(symptoms, domain, self.database)
        else:
            QMessageBox.warning(self, "Input Missing", "Please upload an image or describe symptoms.")
            return

        if result:
            stages_str = "\n".join([f"  • {k}: {v}" for k, v in result.get("stages", {}).items()])
            out = (
                f"--- DIAGNOSIS ---\n"
                f"Confidence: {confidence:.1%}\n"
                f"Disease Name: {result['name']}\n\n"
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
            tab.result_display.setPlainText("No diagnosis could be made. The database may be empty for this domain or the input was not specific enough.")

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
