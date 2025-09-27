from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QComboBox, QLineEdit, QTextEdit, QDialogButtonBox
)

class AddNewDiseaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Disease Information")
        self.layout = QFormLayout(self)
        self.setMinimumWidth(400)

        self.domain_box = QComboBox()
        self.domain_box.addItems(["Plant", "Human", "Animal"])
        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        self.stages_edit = QTextEdit()
        self.causes_edit = QTextEdit()
        self.risk_factors_edit = QTextEdit()
        self.preventive_measures_edit = QTextEdit()
        self.solution_edit = QTextEdit()

        self.layout.addRow("Domain:", self.domain_box)
        self.layout.addRow("Disease Name:", self.name_edit)
        self.layout.addRow("Description:", self.desc_edit)
        self.layout.addRow("Stages (key: value per line):", self.stages_edit)
        self.layout.addRow("Causes:", self.causes_edit)
        self.layout.addRow("Risk Factors:", self.risk_factors_edit)
        self.layout.addRow("Preventive Measures:", self.preventive_measures_edit)
        self.layout.addRow("Solution/Cure:", self.solution_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def get_data(self):
        stages_text = self.stages_edit.toPlainText().strip()
        stages = {}
        lines = [line for line in stages_text.splitlines() if line.strip()]

        if any(":" in line for line in lines):
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    stages[k.strip()] = v.strip()
        elif lines:
             stages["General"] = "\n".join(lines)

        return {
            "domain": self.domain_box.currentText(),
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "stages": stages if stages else {"Info": "Not specified"},
            "causes": self.causes_edit.toPlainText().strip(),
            "risk_factors": self.risk_factors_edit.toPlainText().strip(),
            "preventive_measures": self.preventive_measures_edit.toPlainText().strip(),
            "solution": self.solution_edit.toPlainText().strip(),
        }
