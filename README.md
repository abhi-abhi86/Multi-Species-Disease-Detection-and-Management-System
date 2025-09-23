AI-Powered Multi-Species Disease Detection and Management System
This repository contains a complete PyQt6 application for scanning, diagnosing, and managing diseases across plants, humans, and animals. The system is enhanced with Wikipedia integration for detailed disease information and includes an interactive chatbot for user assistance.

Features
Multi-Domain Support: Separate, organized tabs for Plants, Humans, and Animals.

Symptom Analysis: A keyword-based algorithm provides a likely disease match from user-described symptoms.

Wikipedia Integration: Automatically fetches and displays a summary from Wikipedia for any diagnosed disease.

Interactive Chatbot: A simple chatbot to answer basic questions about diseases stored in the database.

Add New Diseases: A user-friendly dialog to expand the local disease database.

Persistent Storage: All data is saved in data/disease_database.json.

Directory Structure
/DiseaseDetectionApp
├── main.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── add_disease_dialog.py
│   └── chatbot_dialog.py
├── core/
│   ├── __init__.py
│   ├── data_handler.py
│   ├── ml_processor.py
│   └── wikipedia_integration.py
└── data/
    └── disease_database.json

Installation
Clone this repository:

git clone [https://github.com/abhi-abhi86/put.git](https://github.com/abhi-abhi86/put.git)

Install dependencies from the requirements.txt file:

pip install -r requirements.txt

Run the application:

python DiseaseDetectionApp/main.py

Usage
Diagnose: Select a tab, upload an image, or enter symptoms, then click "Diagnose." The results will include a summary from Wikipedia.

Add Disease: Use the "File" menu → "Add New Disease..." to add more diseases.

Chatbot: Access the chatbot from the "File" menu to ask questions.

License
MIT License
