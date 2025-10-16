# DiseaseDetectionApp/ui/add_disease_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QLineEdit, QTextEdit, QDialogButtonBox,
    QVBoxLayout
)
import re

class AddNewDiseaseDialog(QDialog):
    """
    A dialog for adding new disease information to the application's database.
    It saves the data as a new, clean JSON file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Disease Information")
        self.setMinimumWidth(500)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # --- Form Fields ---
        self.domain_box = QComboBox()
        self.domain_box.addItems(["Plant", "Human", "Animal"])

        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        
        self.stages_edit = QTextEdit()
        self.stages_edit.setPlaceholderText("Format: One stage per line, like 'Key: Value'\nExample:\nEarly: Small, irregular spots.\nAdvanced: Lesions enlarge.")
        
        self.causes_edit = QTextEdit()
        self.solution_edit = QTextEdit()
        self.preventive_measures_edit = QTextEdit()
        
        self.image_url_edit = QLineEdit()
        self.image_url_edit.setPlaceholderText("Relative path, e.g., diseases/plant/new_disease_img.jpg")

        # --- Add Widgets to Form Layout ---
        form_layout.addRow("Domain:", self.domain_box)
        form_layout.addRow("Disease Name:", self.name_edit)
        form_layout.addRow("Description:", self.desc_edit)
        form_layout.addRow("Stages (key: value per line):", self.stages_edit)
        form_layout.addRow("Common Causes:", self.causes_edit)
        form_layout.addRow("Solution / Cure:", self.solution_edit)
        form_layout.addRow("Preventive Measures:", self.preventive_measures_edit)
        form_layout.addRow("Reference Image Path:", self.image_url_edit)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # --- Add Form and Buttons to Main Layout ---
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)

    def get_data(self):
        """
        Retrieves, cleans, and formats the data from the form fields into a dictionary.
        Includes improved, more robust parsing for the 'stages' field.
        """
        stages_text = self.stages_edit.toPlainText().strip()
        stages = {}
        
        # Process each non-empty line to extract key-value pairs for stages.
        for line in stages_text.splitlines():
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value: # Ensure both key and value are present
                    stages[key] = value
        
        # If no valid key:value pairs were found but text exists, save it under a general 'Info' key.
        if not stages and stages_text:
             stages["Info"] = stages_text

        # Return a clean dictionary with stripped text.
        return {
            "domain": self.domain_box.currentText(),
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "stages": stages,
            "causes": self.causes_edit.toPlainText().strip(),
            "solution": self.solution_edit.toPlainText().strip(),
            "preventive_measures": self.preventive_measures_edit.toPlainText().strip(),
            # Standardize to use 'image_url' and handle backslashes for path consistency.
            "image_url": self.image_url_edit.text().strip().replace("\\", "/") or None,
        }
