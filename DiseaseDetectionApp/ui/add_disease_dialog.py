from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QLineEdit, QTextEdit, QDialogButtonBox
)

class AddNewDiseaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Disease Information")
        self.layout = QFormLayout(self)

        self.domain_box = QComboBox()
        self.domain_box.addItems(["Plant", "Human", "Animal"])
        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        self.stages_edit = QTextEdit()
        self.solution_edit = QTextEdit()

        self.layout.addRow("Domain:", self.domain_box)
        self.layout.addRow("Disease Name:", self.name_edit)
        self.layout.addRow("Description:", self.desc_edit)
        self.layout.addRow("Stages (Provide details for each stage):", self.stages_edit)
        self.layout.addRow("Solution/Cure:", self.solution_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_data(self):
        stages_text = self.stages_edit.toPlainText().strip()
        stages = {}
        lines = [line for line in stages_text.splitlines() if line.strip()]

        # If any line contains a colon, assume key-value pairs
        if any(":" in line for line in lines):
            for i, line in enumerate(lines):
                if ":" in line:
                    k, v = line.split(":", 1)
                    stages[k.strip()] = v.strip()
                else:
                    stages[f"Stage {i+1}"] = line.strip()
        elif lines:
             stages["General"] = "\n".join(lines)


        return {
            "domain": self.domain_box.currentText(),
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "stages": stages if stages else {"Info": "Not specified"},
            "solution": self.solution_edit.toPlainText().strip(),
        }
