# AI-Powered Multi-Species Disease Detection and Management System

This repository contains a complete PyQt6 application for scanning, diagnosing, and managing diseases across plants, humans, and animals. Disease data is persisted and instantly available for AI-powered diagnosis and management.

## Features

- **Scan Disease:** Upload an image and get instant mock diagnosis.
- **Symptom Analysis:** Type symptoms to get a likely disease match.
- **Add New Diseases:** Use the dialog to add new disease info to the database.
- **Tabbed UI:** Separate tabs for Plants, Humans, and Animals.
- **Persistent Storage:** All data is saved in `data/disease_database.json`.
- **Modular Codebase:** Organized for scalability and easy extension.
- **ML/AI Mocking:** Diagnosis is simulated (random selection) for easy future upgrade.

## Directory Structure

```
/DiseaseDetectionApp
├── main.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   └── add_disease_dialog.py
├── core/
│   ├── __init__.py
│   ├── data_handler.py
│   └── ml_processor.py
└── data/
    └── disease_database.json
```

## Installation

1. Clone this repository:
    ```
    git clone https://github.com/abhi-abhi86/put.git
    ```
2. Install dependencies:
    ```
    pip install PyQt6 Pillow
    ```
3. Run the application:
    ```
    python DiseaseDetectionApp/main.py
    ```

## Usage

- **Diagnose:** Select a tab, upload an image ("Scan Disease"), or enter symptoms, then click "Diagnose".
- **Add Disease:** Use the "File" menu → "Add New Disease..." to add more diseases to the system.
- **Persistent Data:** All diseases added are saved and available for instant diagnosis without restarting.

## Notes

- The ML/AI backend is currently mocked—results are randomly selected from the database for demonstration.
- You can extend the `/core/ml_processor.py` for real ML or AI integration.

## License

MIT License

---
