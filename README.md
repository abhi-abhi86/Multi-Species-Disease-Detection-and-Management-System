![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=35&pause=1000&color=3BF7FF&center=true&vCenter=true&width=600&lines=Hi%2C+I'm+abhi-abhi86;AI+Enthusiast+%7C+Python+Dev+%7C+ML+Researcher)

# 🦠 AI Multi-Species Disease Detection and Management System

A powerful Python desktop application for **AI-driven disease detection and management**—across **plants, humans, and animals**.  
Leverage a pre-trained machine learning model to assist with preliminary diagnosis, and empower your research or field work with data-driven insights.

---

## 🚀 Key Information

- **Tech Stack:** Python | PyQt6 | PyTorch (MobileNetV2) | ReportLab
- **Status:** Actively Developed
- **License:** MIT

---

## ✨ Features

- **Multi-Domain Diagnosis:** Instantly switch between dedicated tabs for Plants, Humans, and Animals. 🌳🧑‍🤝‍🧑🐅
- **Dual Input Modes:** Diagnose with either image uploads (drag-and-drop supported) or via text-based symptom descriptions.
- **AI-Powered Image Analysis:** Uses a trained MobileNetV2 model to match images to known diseases.
- **Symptom-Based Analysis:** Robust keyword-matching for your symptom entries against our extensive database.
- **Comprehensive Results:** Detailed diagnosis with confidence levels, stages, causes, prevention, and recommended solutions. 📄
- **External Data Integration:** Pulls summaries from Wikipedia and the latest research abstracts from NCBI PubMed for a holistic view.
- **Extensible Database:** Add new disease entries through a user-friendly form.
- **PDF Report Generation:** Save or share professional PDF reports for any diagnosis.
- **Location Logging:** Track outbreaks by logging diagnosis locations—viewable on a map. 🗺️
- **Simple Chatbot:** Get quick disease info from our database with a built-in chatbot. 🤖

---

## 🗂️ File Structure Diagram

Visual overview of the core project architecture:

```
/Multi-Species-Disease-Detection-and-Management-System
|
├── DiseaseDetectionApp/
│   ├── core/
│   │   ├── data_handler.py        # 🧠 Loads & saves disease data (JSON).
│   │   ├── ml_processor.py        # 🤖 AI model loading & image prediction.
│   │   ├── worker.py              # ⚙️ Async diagnosis processing.
│   │   ├── wikipedia_integration.py # 🌐 Wikipedia summary fetch.
│   │   ├── ncbi_integration.py      # 🔬 PubMed research fetch.
│   │   └── report_generator.py      # 📄 PDF report creation.
│   │
│   ├── data/
│   │   └── disease_database.json  # (Legacy) Fallback database.
│   │
│   ├── diseases/                  # ★ Main disease info database.
│   │   ├── animal/
│   │   │   └── sarcoptic_mange/
│   │   │       ├── images/        # 🖼️ Training images.
│   │   │       └── sarcoptic_mange.json # 📝 Disease info.
│   │   ├── human/
│   │   └── plant/
│   │
│   ├── ui/
│   │   ├── main_window.py         # 🖥️ Main UI.
│   │   ├── add_disease_dialog.py  # ➕ Add new disease form.
│   │   ├── chatbot_dialog.py      # 💬 Chatbot UI.
│   │   └── ... (other dialogs)
│   │
│   ├── dataset/                   # (Generated) Training images.
│   ├── class_to_name.json         # (Generated) Label mapping.
│   ├── disease_model.pt           # (Generated) Trained model.
│   └── main.py                    # ▶️ App entry point.
│
├── train_disease_classifier.py    # 🚂 Train the AI model.
├── predict_disease.py             # ✅ Command-line prediction script.
├── requirements.txt               # 📦 Python dependencies.
└── README.md                      # 📖 You're here!
```

---

## 🧭 How It Works: Process Flow

1. **Initialization:**  
   - Launch `main.py` to start the app.  
   - Loads disease data and the AI model (`disease_model.pt`). Warns if model files are missing.

2. **User Input:**  
   - Use the UI to upload an image or enter symptoms.  
   - Choose the species domain (Plant, Human, Animal) and click "Diagnose".

3. **Background Processing:**  
   - Diagnosis runs in a background thread (`worker.py`) for a smooth UI.  
   - **If image:** AI model predicts the disease.
   - **If symptoms:** Fuzzy string match finds the closest disease entry.

4. **Data Aggregation:**  
   - Fetches full disease info from the database.  
   - Calls Wikipedia and PubMed integrations for summaries and research.

5. **Results Display:**  
   - UI presents all findings: disease info, confidence, Wikipedia, research abstracts.

6. **Report Generation (Optional):**  
   - Generate a detailed PDF with all results and images via the "Save Report" feature.

This modular structure ensures that the UI, core logic, and data are all separated, making the system easier to maintain and expand.

---

## 🖥️ Installation & Usage

1. **Clone the Repository**
    ```sh
    git clone https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git
    cd Multi-Species-Disease-Detection-and-Management-System
    ```

2. **Create a Virtual Environment**
    - **Windows:**
      ```sh
      python -m venv venv
      venv\Scripts\activate
      ```
    - **macOS/Linux:**
      ```sh
      python3 -m venv venv
      source venv/bin/activate
      ```

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **[Optional] Set Up Google API Credentials**
    - For enhanced Wikipedia/online lookup.  
    - [Instructions here](https://console.cloud.google.com/) (see original file for details).

5. **Prepare the Image Dataset**
    - Organize training images in `DiseaseDetectionApp/dataset/` as:
      ```
      dataset/
        ├── disease_one/
        │   ├── img1.jpg
        └── disease_two/
            ├── img2.jpg
      ```

6. **Train the AI Model**
    ```sh
    python DiseaseDetectionApp/train_disease_classifier.py
    ```

7. **Run the Application**
    ```sh
    python DiseaseDetectionApp/main.py
    ```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> _Crafted with ❤️ by [abhi-abhi86](https://github.com/abhi-abhi86) — AI Enthusiast, Python Developer, ML Researcher._
