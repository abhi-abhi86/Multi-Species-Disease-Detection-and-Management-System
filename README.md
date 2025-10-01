AI Multi-Species Disease Detection and Management System
This project is a desktop application developed in Python that uses a pre-trained machine learning model to assist in the detection and management of diseases across different species (plants, humans, and animals).

Description
The application provides a user-friendly interface built with PyQt6 where users can either upload an image of an affected specimen or describe its symptoms. The system then uses an AI model (MobileNetV2) for image-based analysis or a keyword-matching algorithm for symptom-based analysis to provide a potential diagnosis, detailed information from a local database, and a summary from Wikipedia.

## Contributors

This project is made possible by the contributions of the following individuals:

| Contributor | Commits | Contribution % |
|-------------|---------|----------------|
| ABHISHEK M G | 1 | 50.0% |
| copilot-swe-agent[bot] | 1 | 50.0% |

**Total Commits:** 2

We appreciate all contributions to this project! 🎉

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

Setup and Installation
1. Clone the Repository
git clone [https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git](https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git)
cd Multi-Species-Disease-Detection-and-Management-System

2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Set Up Google API Credentials (Optional)
The application can use the Google Custom Search API to fetch images and summaries online. This is optional but enhances functionality.

Create a Google Custom Search Engine (CSE):

Go to the Custom Search Engine control panel.

Create a new search engine. You can enable "Search the entire web".

Once created, find your Search engine ID (CX).

Get a Google API Key:

Go to the Google Cloud Console.

Create a new project or select an existing one.

Go to APIs & Services > Credentials.

Click Create Credentials > API key.

Important: Restrict your API key to only be used for the "Custom Search API".

Set Environment Variables:
You need to set the API Key and CSE ID as environment variables so the application can access them securely.

Windows (Command Prompt):

setx GOOGLE_API_KEY "YOUR_API_KEY"
setx GOOGLE_CSE_ID "YOUR_CSE_ID"

(You may need to restart your terminal for these to take effect).

Windows (PowerShell):

$env:GOOGLE_API_KEY="YOUR_API_KEY"
$env:GOOGLE_CSE_ID="YOUR_CSE_ID"

macOS/Linux:

export GOOGLE_API_KEY="YOUR_API_KEY"
export GOOGLE_CSE_ID="YOUR_CSE_ID"

(Add these lines to your ~/.bashrc, ~/.zshrc, or shell configuration file to make them permanent).

5. Prepare the Image Dataset
The AI model needs to be trained on a dataset of images. The training script expects the images to be organized in the following structure:

Multi-Species-Disease-Detection-and-Management-System/
└── DiseaseDetectionApp/
    └── dataset/
        ├── disease_one/
        │   ├── image001.jpg
        │   └── image002.png
        ├── disease_two/
        │   ├── image003.jpg
        │   └── image004.jpeg
        └── ...

I have created the dataset directory and moved the existing images into it for you. You can add more images to these folders or create new folders for other diseases. The folder names should be lowercase and use underscores instead of spaces (e.g., rose_black_spot).

6. Train the AI Model
Before running the main application, you must train the model on your dataset.

python DiseaseDetectionApp/train_disease_classifier.py

This script will:

Read the images from the DiseaseDetectionApp/dataset directory.

Train the MobileNetV2 model.

Create two essential files in the DiseaseDetectionApp directory:

disease_model.pt: The trained model weights.

class_to_name.json: A mapping of model output to disease names.

Usage
After you have successfully trained the model, you can run the main application.

python DiseaseDetectionApp/main.py

License
This project is licensed under the MIT License. See the LICENSE file for details.
