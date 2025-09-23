from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QTextEdit, QMessageBox, QFileDialog, QMenuBar, QMenu
)
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import Qt
from ui.add_disease_dialog import AddNewDiseaseDialog
from core.data_handler import load_database, save_disease
from core.ml_processor import predict_from_image, predict_from_symptoms

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Multi-Species Disease Management System")
        self.resize(800, 600)
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
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(add_action)
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)
        self.setMenuBar(menubar)

    def create_domain_tab(self, domain_name):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Input Section
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()

        image_label = QLabel("Upload an Image")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setFixedSize(256, 256)
        image_label.setStyleSheet("border:1px solid gray;")
        upload_btn = QPushButton("Scan Disease (Upload Image)")
        symptom_input = QTextEdit()
        symptom_input.setPlaceholderText("Or describe the symptoms here...")
        diagnose_btn = QPushButton("Diagnose")

        input_layout.addWidget(image_label, 0, 0, 2, 1)
        input_layout.addWidget(upload_btn, 2, 0)
        input_layout.addWidget(QLabel("Symptoms Description:"), 0, 1)
        input_layout.addWidget(symptom_input, 1, 1)
        input_layout.addWidget(diagnose_btn, 2, 1)
        input_group.setLayout(input_layout)

        # Output Section
        result_group = QGroupBox("Diagnosis Results")
        result_layout = QVBoxLayout()
        result_display = QTextEdit()
        result_display.setReadOnly(True)
        result_layout.addWidget(result_display)
        result_group.setLayout(result_layout)

        main_layout.addWidget(input_group)
        main_layout.addWidget(result_group)
        main_widget.setLayout(main_layout)

        # Store widgets for later access
        main_widget.image_label = image_label
        main_widget.symptom_input = symptom_input
        main_widget.result_display = result_display
        main_widget.domain = domain_name

        upload_btn.clicked.connect(lambda: self.upload_image(domain_name))
        diagnose_btn.clicked.connect(lambda: self.run_diagnosis(domain_name))
        return main_widget

    def get_active_domain(self):
        idx = self.tab_widget.currentIndex()
        return ["Plant", "Human", "Animal"][idx]

    def upload_image(self, domain):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            tab = self.domain_tabs[domain]
            pixmap = QPixmap(file_path)
            tab.image_label.setPixmap(pixmap.scaled(tab.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            self.current_image_paths[domain] = file_path

    def run_diagnosis(self, domain):
        tab = self.domain_tabs[domain]
        image_path = self.current_image_paths[domain]
        symptoms = tab.symptom_input.toPlainText().strip()
        if image_path:
            result = predict_from_image(image_path, domain, self.database)
        elif symptoms:
            result = predict_from_symptoms(symptoms, domain, self.database)
        else:
            tab.result_display.setPlainText("Please provide an image or symptoms description for diagnosis.")
            return

        if result:
            stages_str = "\n".join([f"{k}: {v}" for k, v in result.get("stages", {}).items()])
            out = (
                f"Disease Name: {result['name']}\n"
                f"Description: {result['description']}\n"
                f"Stages:\n{stages_str}\n"
                f"Solution:\n{result['solution']}"
            )
            tab.result_display.setPlainText(out)
        else:
            tab.result_display.setPlainText("No diagnosis could be made. Try adding more diseases or refining input.")

    def open_add_disease_dialog(self):
        dialog = AddNewDiseaseDialog(self)
        result = dialog.exec()
        if result == dialog.DialogCode.Accepted:
            data = dialog.get_data()
            save_disease(data)
            self.database = load_database()
            QMessageBox.information(self, "Success", "Disease information saved and available for diagnosis.")