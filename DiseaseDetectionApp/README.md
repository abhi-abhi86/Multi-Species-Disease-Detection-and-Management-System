# Multi-Species Disease Detection and Management System

An AI-powered desktop application for detecting and managing diseases across plants, humans, and animals using advanced machine learning and computer vision technologies.

## 🚀 Features

### Core Functionality
- **Multi-Species Detection**: Supports disease detection for plants, humans, and animals
- **AI-Powered Diagnosis**: Machine learning model trained on 11+ disease types
- **Real-time Analysis**: Instant diagnosis with confidence scores
- **Research Integration**: PubMed and Wikipedia integration for disease information
- **Professional UI**: Modern PyQt5 interface with fade-in animations

### Advanced Features
- **Disease Database**: Expandable database with 13+ pre-loaded diseases
- **Add New Diseases**: User-friendly interface to add custom diseases
- **Model Retraining**: Intelligent retraining workflow for new diseases
- **Comprehensive Testing**: Built-in test suite for validation
- **Developer Tools**: Access to system information and diagnostics

## 📊 Supported Diseases

### 🌱 Plants (4 diseases)
- Powdery Mildew
- Citrus Canker
- Rose Black Spot
- Areca Nut Disease

### 👤 Humans (4 diseases)
- Acne Vulgaris
- Eczema
- Smoker's Lung
- AIDS

### 🐾 Animals (3 diseases)
- Lumpy Skin Disease
- Sarcoptic Mange
- Swine Erysipelas

## 🛠️ Installation

### Option 1: Run from Source (Development)
```bash
# Clone the repository
git clone https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System.git
cd Multi-Species-Disease-Detection-and-Management-System/DiseaseDetectionApp

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Option 2: Standalone macOS Application
Download the latest release from the [Releases](https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System/releases) page.

## 🏗️ Building Standalone Application

### For macOS
```bash
# Install py2app
pip install py2app

# Build the application
python setup.py py2app

# The app will be created in the 'dist' folder
```

### For Windows (using PyInstaller)
```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller --onefile --windowed main.py

# The executable will be in the 'dist' folder
```

## 🎯 Usage

1. **Launch the Application**: Run `main.py` or the standalone app
2. **Select Species Tab**: Choose between Plants, Humans, or Animals
3. **Upload Image**: Click "Browse Image" to select a disease image
4. **Get Diagnosis**: Click "Diagnose" for AI-powered analysis
5. **View Results**: See diagnosis, confidence score, and treatment recommendations
6. **Research More**: Access PubMed articles and Wikipedia information

### Adding New Diseases
1. Go to **File → Add New Disease** (Ctrl+N)
2. Fill in disease details (name, description, symptoms, treatment)
3. Optionally retrain the model immediately or later
4. Use **Tools → Retrain Model** (Ctrl+R) to include new diseases

## 🔧 System Requirements

- **OS**: macOS 10.14+, Windows 10+, Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for application + models
- **Dependencies**: PyQt5, PyTorch, OpenCV

## 🧪 Testing

Run the comprehensive test suite:
```bash
python comprehensive_test.py
```

Run specific tests:
```bash
python test_disease_detection.py  # Disease detection tests
python test_train_model.py        # Model training tests
```

## 📁 Project Structure

```
DiseaseDetectionApp/
├── main.py                     # Main application entry point
├── core/                       # Core business logic
│   ├── ml_processor.py         # Machine learning processor
│   ├── worker.py               # Background worker threads
│   ├── data_handler.py         # Database and data management
│   ├── wikipedia_integration.py # Wikipedia API integration
│   ├── ncbi_integration.py     # PubMed/NCBI integration
│   └── retraining_worker.py    # Model retraining worker
├── ui/                         # User interface components
│   ├── main_window.py          # Main application window
│   ├── add_disease_dialog.py   # Add disease dialog
│   └── create_spinner.py       # Loading spinner component
├── diseases/                   # Disease database
│   ├── plant/                  # Plant diseases
│   ├── human/                  # Human diseases
│   └── animal/                 # Animal diseases
├── train_disease_classifier.py # Model training script
├── predict_disease.py          # Disease prediction script
├── comprehensive_test.py       # Comprehensive test suite
├── setup.py                    # Py2app setup script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Abhishek MG**
- GitHub: [@abhi-abhi86](https://github.com/abhi-abhi86)
- Email: abhishekmgabhishekmg726@gmail.com

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- PyQt5 community for the GUI framework
- NCBI/PubMed for research data access
- Wikipedia for general disease information

---

**Version**: 1.0.0
**Last Updated**: 2025
**Status**: Production Ready
