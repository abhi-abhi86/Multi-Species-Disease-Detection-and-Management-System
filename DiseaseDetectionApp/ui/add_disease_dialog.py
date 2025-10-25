                                              
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QLineEdit, QTextEdit, QDialogButtonBox,
    QVBoxLayout, QFileDialog
)
import re
import os

class AddNewDiseaseDialog(QDialog):
    """
    A dialog for adding new disease information to the application's database.
    It saves the data as a new, clean JSON file.
    """
    def __init__(self, parent=None, prefilled_data=None, image_path=None, domain=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Disease Information")
        self.setMinimumWidth(500)

                             
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

                             
        self.domain_box = QComboBox()
        self.domain_box.addItems(["Plant", "Human", "Animal"])

        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()

        self.stages_edit = QTextEdit()
        self.stages_edit.setPlaceholderText("Format: One stage per line, like 'Key: Value'\nExample:\nAcute: Sudden onset.\nChronic: Long-term.")

        self.causes_edit = QTextEdit()
        self.risk_factors_edit = QTextEdit()
        self.solution_edit = QTextEdit()
        self.preventive_measures_edit = QTextEdit()

        self.image_url_edit = QLineEdit()
        self.image_url_edit.setPlaceholderText("Relative path, e.g., user_added_diseases/plant/new_disease_img.jpg")

        self.browse_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Open)
        self.browse_button.clicked.connect(self.browse_image)

                                            
        form_layout.addRow("Domain:", self.domain_box)
        form_layout.addRow("Disease Name:", self.name_edit)
        form_layout.addRow("Description:", self.desc_edit)
        form_layout.addRow("Stages (key: value per line):", self.stages_edit)
        form_layout.addRow("Common Causes:", self.causes_edit)
        form_layout.addRow("Risk Factors:", self.risk_factors_edit)
        form_layout.addRow("Solution / Cure:", self.solution_edit)
        form_layout.addRow("Preventive Measures:", self.preventive_measures_edit)
        form_layout.addRow("Reference Image Path:", self.image_url_edit)
        form_layout.addRow("Browse Image:", self.browse_button)

                                
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

                                                     
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)

                                          
        if prefilled_data:
            self.prefill_data(prefilled_data)

                                                            
        if image_path:
            self.image_url_edit.setText(image_path)
        if domain:
            index = self.domain_box.findText(domain)
            if index >= 0:
                self.domain_box.setCurrentIndex(index)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_url_edit.setText(file_path)

    def prefill_data(self, data):
        if 'domain' in data:
            index = self.domain_box.findText(data['domain'])
            if index >= 0:
                self.domain_box.setCurrentIndex(index)
        if 'name' in data:
            self.name_edit.setText(data['name'])
        if 'description' in data:
            self.desc_edit.setText(data['description'])
        if 'stages' in data:
            stages_text = "\n".join([f"{k}: {v}" for k, v in data['stages'].items()])
            self.stages_edit.setText(stages_text)
        if 'causes' in data:
            self.causes_edit.setText(data['causes'])
        if 'risk_factors' in data:
            self.risk_factors_edit.setText(data['risk_factors'])
        if 'solution' in data:
            self.solution_edit.setText(data['solution'])
        if 'preventive_measures' in data:
            self.preventive_measures_edit.setText(data['preventive_measures'])
        if 'image_path' in data:
            self.image_url_edit.setText(data['image_path'])

    def get_data(self):
        """
        Retrieves, cleans, and formats the data from the form fields into a dictionary.
        Includes improved, more robust parsing for the 'stages' field.
        """
        stages_text = self.stages_edit.toPlainText().strip()
        stages = {}
        
                                                                            
        for line in stages_text.splitlines():
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:                                        
                    stages[key] = value
        
                                                                                                     
        if not stages and stages_text:
             stages["Info"] = stages_text

                                                       
        return {
            "domain": self.domain_box.currentText(),
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "stages": stages,
            "causes": self.causes_edit.toPlainText().strip(),
            "risk_factors": self.risk_factors_edit.toPlainText().strip(),
            "solution": self.solution_edit.toPlainText().strip(),
            "preventive_measures": self.preventive_measures_edit.toPlainText().strip(),
                                                                                         
            "image_url": self.image_url_edit.text().strip().replace("\\", "/") or None,
        }


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = AddNewDiseaseDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        data = dialog.get_data()
        print("Disease data:", data)
                                                                                    
    sys.exit(app.exec())
