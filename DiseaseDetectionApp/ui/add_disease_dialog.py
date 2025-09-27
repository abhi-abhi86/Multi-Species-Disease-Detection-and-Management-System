from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QLineEdit, QTextEdit, QDialogButtonBox
)

class AddNewDiseaseDialog(QDialog):
    """
    A dialog for adding new disease information to the database.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Disease Information")
        self.layout = QFormLayout(self)
        self.setMinimumWidth(450) # Increased width for better layout

        # --- UI Components ---
        self.domain_box = QComboBox()
        self.domain_box.addItems(["Plant", "Human", "Animal"])

        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        self.stages_edit = QTextEdit()
        self.stages_edit.setPlaceholderText("Example:\nEarly: Small, irregular spots.\nAdvanced: Lesions enlarge.")
        self.causes_edit = QTextEdit()
        self.risk_factors_edit = QTextEdit()
        self.preventive_measures_edit = QTextEdit()
        self.solution_edit = QTextEdit()
        self.image_url_edit = QLineEdit()
        self.image_url_edit.setPlaceholderText("e.g., images/My_Disease.jpg")

        # --- Layout ---
        self.layout.addRow("Domain:", self.domain_box)
        self.layout.addRow("Disease Name:", self.name_edit)
        self.layout.addRow("Description:", self.desc_edit)
        self.layout.addRow("Stages (key: value per line):", self.stages_edit)
        self.layout.addRow("Causes:", self.causes_edit)
        self.layout.addRow("Risk Factors:", self.risk_factors_edit)
        self.layout.addRow("Preventive Measures:", self.preventive_measures_edit)
        self.layout.addRow("Solution/Cure:", self.solution_edit)
        self.layout.addRow("Image File Path (optional):", self.image_url_edit)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_data(self):
        """
        Retrieves and formats the data from the form fields.
        Improved stage parsing to be more robust.
        """
        stages_text = self.stages_edit.toPlainText().strip()
        stages = {}
        # Process lines that are not empty
        lines = [line.strip() for line in stages_text.splitlines() if line.strip()]

        # Check if any line has the key:value format
        if any(":" in line for line in lines):
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value: # Ensure both key and value are present
                        stages[key] = value
        # If no colons are found, but there is text, save it under a general info key
        elif lines:
             stages["Info"] = "\n".join(lines)

        return {
            "domain": self.domain_box.currentText(),
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            # If no stages were parsed, provide a default value
            "stages": stages if stages else {"Info": "Not specified"},
            "causes": self.causes_edit.toPlainText().strip(),
            "risk_factors": self.risk_factors_edit.toPlainText().strip(),
            "preventive_measures": self.preventive_measures_edit.toPlainText().strip(),
            "solution": self.solution_edit.toPlainText().strip(),
            "image_url": self.image_url_edit.text().strip() or None, # Store image path
        }
