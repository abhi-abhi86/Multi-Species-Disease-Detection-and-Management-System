AI Multi-Species Disease Detection and Management System
This project is a desktop application developed in Python that uses a pre-trained machine learning model to assist in the detection and management of diseases across different species (plants, humans, and animals).

Description
The application provides a user-friendly interface built with PyQt6 where users can either upload an image of an affected specimen or describe its symptoms. The system then uses an AI model (MobileNetV2) for image-based analysis or a keyword-matching algorithm for symptom-based analysis to provide a potential diagnosis, detailed information from a local database, and a summary from Wikipedia.

Features
Multi-Domain Diagnosis: Supports separate tabs for Plants, Humans, and Animals.

Dual Input Modes: Diagnose diseases using either image uploads (with drag-and-drop) or text-based symptom descriptions.

AI-Powered Image Analysis: Utilizes a pre-trained MobileNetV2 model from PyTorch to identify features in images and match them to known diseases in the database.

Symptom-Based Analysis: A robust keyword-matching algorithm that compares user-input symptoms against the disease database.

Comprehensive Results: Displays diagnosis confidence, disease information, known stages, causes, preventive measures, and recommended solutions.

Wikipedia Integration: Fetches and displays a concise summary from Wikipedia for the diagnosed disease.

Extensible Database: Users can add new disease information to the local JSON database through a dedicated form.

Location Logging: Option to log the geographical location of a diagnosis to help track disease outbreaks.

Simple Chatbot: An informational chatbot to quickly look up disease descriptions from the database.

Installation
Clone the repository:

git clone [https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git](https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git)

Navigate to the project directory:

cd Multi-Species-Disease-Detection-and-Management-System

Install the required dependencies:

pip install -r requirements.txt

Usage
Navigate to the application's root directory and run the main.py script:

python DiseaseDetectionApp/main.py

License
This project is licensed under the MIT License.
