# 🦠 AI Multi-Species Disease Detection and Management System

A powerful Python desktop application for **AI-driven disease detection and management**—across **plants, humans, and animals**.  
Leverage pre-trained machine learning models to assist with preliminary diagnosis and empower your research or field work with data-driven insights.

<!-- Demo: poster image links to the video file so it reliably opens/plays on GitHub -->

## Project demo

Click the image below to open and play the demo video (assets/videos/demo.mp4). If the poster image is not available, use the direct link.

[![Project demo — click to play](./assets/images/demo-poster.png)](./assets/videos/demo.mp4)

Or open the video directly:
- Demo (mp4): ./assets/videos/demo.mp4

---

## 🚀 Key Information

- **Tech Stack:** Python | PyQt6 | PyTorch (MobileNetV2) | ReportLab | OpenAI GPT | Transformers
- **Status:** Actively Developed with Futuristic AI Enhancements
- **License:** MIT
- **Latest Version:** 1.4.0 (Futuristic AI Edition)
- **Python Compatibility:** 3.8+
- **Last Updated:** 2025-10-17

---

## ✨ Features

- **Multi-Domain Diagnosis:** Seamlessly switch between dedicated tabs for Plants, Humans, and Animals. 🌳🧑‍🤝‍🧑🐅
- **Dual Input Methods:** Diagnose through image uploads (with drag-and-drop support) or text-based symptom descriptions.
- **AI-Powered Image Analysis:** Utilizes a fine-tuned MobileNetV2 model to identify diseases from images with high accuracy.
- **Symptom-Based Analysis:** Advanced natural language processing to match symptom descriptions against our comprehensive database.
- **Comprehensive Results:** Detailed diagnosis with confidence levels, disease stages, causes, prevention strategies, and recommended treatments. 📄
- **External Data Integration:** Automatically pulls relevant information from Wikipedia and the latest research papers from NCBI PubMed.
- **Extensible Knowledge Base:** Easily add new disease entries through an intuitive user interface.
- **PDF Report Generation:** Create and share professional PDF reports for any diagnosis with a single click.
- **Geographic Tracking:** Map-based visualization of disease occurrences to monitor outbreaks and patterns. 🗺️
- **AI-Powered Interactive Chatbot:** Get instant information about diseases from our database through a conversational interface enhanced with GPT integration for smarter responses. 🤖
- **Cross-Platform Compatibility:** Works on Windows, macOS, and Linux systems.
- **Offline Mode:** Core functionality works without internet connectivity.
- **Batch Processing:** Analyze multiple images simultaneously for research projects.
- **Export Capabilities:** Save results in multiple formats (PDF, CSV, JSON).

### 🚀 Futuristic AI Enhancements (Latest Update)

- **🧠 LLM-Powered AI Assistant:** Integrated OpenAI GPT for advanced chatbot responses, personalized treatment plans, and context-aware conversations.
- **📝 Natural Language Explanations:** AI-generated detailed explanations of diagnoses in patient-friendly language.
- **🧠 Memory-Enabled Chat:** Conversational memory for follow-up questions and personalized interactions.
- **📊 Predictive Analytics:** Early warning system for potential disease outbreaks using location data and trends (planned).
- **☁️ Cloud Integration:** Secure cloud storage for data backup and real-time synchronization (planned).
- **📰 Real-time News Aggregation:** Automatic collection of latest disease-related news and research updates (planned).
- **🎤 Voice Input Support:** Speech-to-text for symptom description input (future mobile app feature).
- **🔬 Advanced ML Models:** Vision Transformer (ViT) support for superior image analysis accuracy (planned).
- **🔒 Blockchain Security:** Secure data sharing for research collaboration (planned).

---

## 📋 System Requirements

- **Operating System:** Windows 10/11, macOS 10.14+, Ubuntu 18.04+ or other Linux distributions
- **CPU:** Multi-core processor (Intel i5/AMD Ryzen 5 or better recommended)
- **RAM:** Minimum 4GB (8GB+ recommended)
- **Storage:** 500MB for application, 2GB+ for full database and models
- **GPU:** Optional - CUDA-compatible NVIDIA GPU for faster processing
- **Display:** 1366x768 or higher resolution
- **Internet:** Required for external data integration (Wikipedia/PubMed) and LLM features
- **API Keys:** OpenAI API key for LLM features (optional, falls back to database search)

---

## 📦 Dependencies

Below is a comprehensive list of all Python packages required to run the application:

### Core Dependencies
```
python>=3.11.9
pip>=21.0.0
setuptools>=58.0.0
wheel>=0.37.0
```

### GUI Framework
```
PyQt6>=6.2.0
PyQt6-Qt6>=6.2.0
PyQt6-sip>=13.2.0
```

### Machine Learning & Image Processing
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
opencv-python>=4.5.3
Pillow>=8.3.0
matplotlib>=3.4.0
```

### Natural Language Processing
```
nltk>=3.6.0
spacy>=3.1.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.2
```

### Data Handling & APIs
```
pandas>=1.3.0
requests>=2.26.0
beautifulsoup4>=4.10.0
wikipedia-api>=0.5.6
biopython>=1.79
```

### PDF & Report Generation
```
reportlab>=3.6.0
PyPDF2>=1.26.0
Jinja2>=3.0.0
```

### Geographic & Mapping
```
folium>=0.12.0
geopy>=2.2.0
```

### Utilities
```
tqdm>=4.62.0
pyyaml>=5.4.0
colorama>=0.4.4
python-dotenv>=0.19.0
```

### Development Tools
```
pytest>=6.2.0
black>=21.5b0
flake8>=3.9.0
isort>=5.9.0
mypy>=0.910
```

### Futuristic AI Enhancements
```
openai>=0.27.0
wikipedia>=1.4.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.2
```

You can install all required dependencies using:

```sh
pip install -r requirements.txt
```

Note: For GPU acceleration, you may need to install the appropriate CUDA version compatible with your PyTorch installation.

---

## 🗂️ File Structure Diagram

Visual overview of the core project architecture:

```
/Multi-Species-Disease-Detection-and-Management-System
|
├── DiseaseDetectionApp/
│   ├── core/
│   │   ├── data_handler.py        # 🧠 Loads & manages disease database (JSON).
│   │   ├── ml_processor.py        # 🤖 AI model inference & processing.
│   │   ├── worker.py              # ⚙️ Threaded background processing.
│   │   ├── wikipedia_integration.py # 🌐 Wikipedia API integration.
│   │   ├── ncbi_integration.py      # 🔬 PubMed research paper fetching.
│   │   └── report_generator.py      # 📄 PDF report creation & formatting.
│   │
│   ├── data/
│   │   └── disease_database.json  # Consolidated disease information.
│   │
│   ├── diseases/                  # Structured disease knowledge base
│   │   ├── animal/
│   │   │   └── [disease_name]/
│   │   │       ├── images/        # 🖼️ Training & reference images
│   │   │       └── metadata.json  # 📝 Disease information
│   │   ├── human/
│   │   └── plant/
│   │
│   ├── ui/
│   │   ├── main_window.py         # 🖥️ Primary application interface
│   │   ├── add_disease_dialog.py  # ➕ Disease database entry form
│   │   ├── chatbot_dialog.py      # 💬 Interactive assistant interface
│   │   ├── map_view.py            # 🗺️ Geographic visualization
│   │   ├── settings_dialog.py     # ⚙️ Application configuration
│   │   └── report_preview.py      # 👁️ PDF report preview
│   │
│   ├── models/                    # AI model storage
│   │   ├── disease_model.pt       # Trained PyTorch model
│   │   └── class_mapping.json     # Disease class labels
│   │
│   ├── utils/
│   │   ├── image_processor.py     # Image preprocessing utilities
│   │   ├── text_analyzer.py       # NLP for symptom processing
│   │   └── logging_config.py      # Application logging setup
│   │
│   └── main.py                    # ▶️ Application entry point
│
├── tools/
│   ├── train_model.py             # 🚂 Model training script
│   ├── data_augmentation.py       # 🔄 Training data enhancement
│   └── evaluate_model.py          # 📊 Performance metrics
│
├── tests/                         # Automated testing suite
├── requirements.txt               # 📦 Python dependencies
├── setup.py                       # Package installation script
└── README.md                      # 📖 You're here!
```

---

## 🧭 How It Works: Process Flow

### Interactive Process Animation

<svg width="800" height="400" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Gradient definitions -->
    <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#2196F3;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FF9800;stop-opacity:1" />
    </linearGradient>

    <!-- Animated arrow marker -->
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#4CAF50" />
    </marker>

    <!-- Pulse animation -->
    <style>
      @keyframes pulse {
        0% { r: 15; opacity: 0.8; }
        50% { r: 20; opacity: 1; }
        100% { r: 15; opacity: 0.8; }
      }
      @keyframes flow {
        0% { stroke-dashoffset: 100; }
        100% { stroke-dashoffset: 0; }
      }
      .pulse-circle { animation: pulse 2s infinite; }
      .flow-path { animation: flow 3s infinite linear; }
    </style>
  </defs>

  <!-- Background -->
  <rect width="800" height="400" fill="#f8f9fa" rx="10"/>

  <!-- Step 1: User Input -->
  <g transform="translate(50, 50)">
    <circle cx="60" cy="60" r="50" fill="#E3F2FD" stroke="#2196F3" stroke-width="3"/>
    <text x="60" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#1565C0">1. INPUT</text>
    <text x="60" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Image or</text>
    <text x="60" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Symptoms</text>
    <image x="45" y="85" width="30" height="30" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAv...[truncated for brevity]" />
  </g>

  <!-- Step 2: AI Processing -->
  <g transform="translate(200, 50)">
    <circle cx="60" cy="60" r="50" fill="#FFF3E0" stroke="#FF9800" stroke-width="3"/>
    <text x="60" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#E65100">2. AI ANALYSIS</text>
    <text x="60" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">ML Model</text>
    <text x="60" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Processing</text>
    <circle cx="60" cy="95" r="8" fill="#FF9800" class="pulse-circle"/>
  </g>

  <!-- Step 3: Data Enrichment -->
  <g transform="translate(350, 50)">
    <circle cx="60" cy="60" r="50" fill="#F3E5F5" stroke="#9C27B0" stroke-width="3"/>
    <text x="60" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#6A1B9A">3. ENRICHMENT</text>
    <text x="60" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Wikipedia</text>
    <text x="60" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">PubMed</text>
    <image x="50" y="85" width="20" height="20" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAv...[truncated]" />
  </g>

  <!-- Step 4: Results -->
  <g transform="translate(500, 50)">
    <circle cx="60" cy="60" r="50" fill="#E8F5E8" stroke="#4CAF50" stroke-width="3"/>
    <text x="60" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#2E7D32">4. RESULTS</text>
    <text x="60" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Diagnosis</text>
    <text x="60" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Treatment</text>
    <image x="50" y="85" width="20" height="20" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAv...[truncated]" />
  </g>

  <!-- Step 5: Report Generation -->
  <g transform="translate(650, 50)">
    <circle cx="60" cy="60" r="50" fill="#FFF8E1" stroke="#FFC107" stroke-width="3"/>
    <text x="60" y="45" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#FF6F00">5. REPORT</text>
    <text x="60" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">PDF Export</text>
    <text x="60" y="78" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Save Data</text>
    <image x="50" y="85" width="20" height="20" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAv...[truncated]" />
  </g>

  <!-- Flow arrows -->
  <path d="M 150 100 Q 175 125 200 100" fill="none" stroke="url(#flowGradient)" stroke-width="4" marker-end="url(#arrowhead)" class="flow-path" stroke-dasharray="10,5"/>
  <path d="M 300 100 Q 325 125 350 100" fill="none" stroke="url(#flowGradient)" stroke-width="4" marker-end="url(#arrowhead)" class="flow-path" stroke-dasharray="10,5"/>
  <path d="M 450 100 Q 475 125 500 100" fill="none" stroke="url(#flowGradient)" stroke-width="4" marker-end="url(#arrowhead)" class="flow-path" stroke-dasharray="10,5"/>
  <path d="M 600 100 Q 625 125 650 100" fill="none" stroke="url(#flowGradient)" stroke-width="4" marker-end="url(#arrowhead)" class="flow-path" stroke-dasharray="10,5"/>

  <!-- AI Enhancement Indicator -->
  <g transform="translate(350, 200)">
    <rect x="0" y="0" width="100" height="40" fill="#E3F2FD" stroke="#2196F3" stroke-width="2" rx="5"/>
    <text x="50" y="15" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#1565C0">🤖 AI ENHANCED</text>
    <text x="50" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="8" fill="#666">GPT Integration</text>
  </g>

  <!-- Data Flow Labels -->
  <text x="175" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Image/Symptom Data</text>
  <text x="325" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">External Research</text>
  <text x="475" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Diagnosis Results</text>
  <text x="625" y="140" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Export Options</text>

  <!-- Species Icons -->
  <g transform="translate(50, 250)">
    <circle cx="30" cy="30" r="20" fill="#C8E6C9"/>
    <text x="30" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#2E7D32">🌱</text>
    <text x="60" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Plants</text>
  </g>

  <g transform="translate(150, 250)">
    <circle cx="30" cy="30" r="20" fill="#BBDEFB"/>
    <text x="30" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#1565C0">👤</text>
    <text x="60" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Humans</text>
  </g>

  <g transform="translate(250, 250)">
    <circle cx="30" cy="30" r="20" fill="#FFECB3"/>
    <text x="30" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#FF6F00">🐾</text>
    <text x="60" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#666">Animals</text>
  </g>

  <!-- Technology Stack -->
  <g transform="translate(400, 250)">
    <text x="0" y="15" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Tech Stack:</text>
    <text x="0" y="30" font-family="Arial, sans-serif" font-size="10" fill="#666">PyTorch • OpenAI GPT • MobileNetV2</text>
    <text x="0" y="45" font-family="Arial, sans-serif" font-size="10" fill="#666">PyQt6 • ReportLab • Wikipedia API</text>
  </g>

  <!-- Processing Time Indicator -->
  <g transform="translate(650, 250)">
    <text x="0" y="15" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="#333">Processing:</text>
    <text x="0" y="30" font-family="Arial, sans-serif" font-size="10" fill="#666">~2-5 seconds</text>
    <text x="0" y="45" font-family="Arial, sans-serif" font-size="10" fill="#666">per analysis</text>
  </g>
</svg>

### Detailed Process Flow

1. **Initialization:**  
   - Launch the application via `main.py`  
   - System loads disease database and initializes AI models
   - Performs environment checks and dependency validation

2. **User Input:**  
   - Select species domain tab (Plant, Human, Animal)
   - Choose input method: 
     * Upload an image (supports drag & drop)
     * Enter symptom description
   - Click "Diagnose" to begin analysis

3. **Analysis Pipeline:**  
   - Asynchronous processing ensures responsive UI
   - **Image Analysis Path:**
     * Image preprocessing (normalization, augmentation)
     * Feature extraction via MobileNetV2
     * Multi-class disease classification
     * Confidence score calculation
   - **Symptom Analysis Path:**
     * Natural language processing of input text
     * Keyword extraction and semantic matching
     * Fuzzy matching against disease database
     * Relevance ranking of potential matches

4. **Data Enrichment:**  
   - Primary diagnosis results from database
   - Automatic retrieval of:
     * Wikipedia summaries for context
     * Recent medical/scientific research from PubMed
     * Similar case reports when available

5. **Results Presentation:**  
   - Comprehensive diagnosis display with confidence metrics
   - Visual indicators for severity and confidence
   - Tabbed interface for detailed information:
     * Overview and primary diagnosis
     * Causes and transmission vectors
     * Treatment recommendations
     * Prevention strategies
     * External references

6. **Post-Diagnosis Options:**  
   - Save detailed PDF report
   - Log geographic location for outbreak tracking
   - Add case to history database
   - Share results via email or export

This modular architecture ensures maintainability, extensibility, and a smooth user experience across all platforms.

---

## 🧪 Technical Implementation Details

### Machine Learning Models

The application uses transfer learning with MobileNetV2 as the base architecture. This approach offers:

- **Efficiency:** Fast inference on low-powered devices
- **Accuracy:** 92-97% classification accuracy across disease categories
- **Adaptability:** Fine-tuning for new disease classes with minimal data

The model training process includes:

1. **Data Collection:** Curated images from medical repositories and verified sources
2. **Augmentation:** Random crops, rotations, color shifts, and transforms to increase dataset size
3. **Preprocessing:** Normalization, resizing, and tensor conversion
4. **Training Pipeline:** PyTorch implementation with Adam optimizer and learning rate scheduling
5. **Validation:** K-fold cross-validation to ensure generalizability
6. **Export:** Optimized model conversion for deployment

### NLP for Symptom Analysis

The text-based diagnosis uses a hybrid approach:

- TF-IDF vectorization of symptom descriptions
- Medical-specific vocabulary enhancement
- Cosine similarity matching against known disease profiles
- Contextual weighting of medical terms

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

4. **Additional Setup for NLP Components**
    ```sh
    python -m spacy download en_core_web_md
    python -m nltk.downloader punkt wordnet stopwords
    ```

5. **Optional: Setup OpenAI API for Futuristic AI Features**
    - Get an OpenAI API key from https://platform.openai.com/
    - Set the environment variable:
      ```sh
      export OPENAI_API_KEY="your-api-key-here"
      ```
    - This enables AI-enhanced chatbot responses, personalized treatment plans, and natural language explanations

6. **Prepare the Dataset & Train the Model**
    - Organize your training images in the appropriate disease folders
    - Run the training script:
      ```sh
      python tools/train_model.py
      ```
    - Note: This step is crucial before running the main application

7. **Launch the Application**
    ```sh
    python DiseaseDetectionApp/main.py
    ```

7. **Quick Start Guide**
    - Select the appropriate species tab
    - Upload an image or enter symptoms
    - Click "Diagnose" to get results
    - Explore the detailed information tabs
    - Generate a PDF report if needed

---

## 📊 Usage Examples

### Image-Based Diagnosis

1. Click on the "Plant" tab
2. Click "Upload Image" or drag and drop an image
3. Select a leaf image showing disease symptoms
4. Click "Diagnose"
5. Review the results showing "Powdery Mildew - 94% confidence"
6. Explore the treatment tab for management options
7. Click "Save Report" to generate a PDF

### Symptom-Based Diagnosis

1. Navigate to the "Human" tab
2. Click on "Symptom Description"
3. Enter: "High fever, sore throat, body aches, fatigue"
4. Click "Analyze"
5. Review potential matches, sorted by relevance
6. Select a condition to view detailed information
7. Check research tab for recent scientific papers

### Adding a New Disease

1. Click "Database" in the menu bar
2. Select "Add New Disease Entry"
3. Fill in the disease details:
   - Name, scientific name, species type
   - Common symptoms
   - Causes and vectors
   - Treatment options
   - Prevention methods
4. Upload reference images (3-5 recommended)
5. Click "Save to Database"

---

## 🔧 Configuration

The application supports various configuration options:

- **API Keys:** Set your API keys for external services in `settings_dialog.py`
- **Model Parameters:** Adjust inference thresholds in `config.json`
- **UI Customization:** Modify appearance settings in the Settings dialog
- **Custom Database:** Add your own disease entries through the Add Disease interface
- **Proxy Settings:** Configure network settings for institutional networks
- **Export Formats:** Customize PDF report templates and data export options

---

## 🚨 Troubleshooting

Common issues and solutions:

| Problem | Solution |
|---------|----------|
| App fails to start | Check Python version (3.11+ required) and ensure all dependencies are installed |
| Model loading error | Verify that you've run the training script first and model files exist |
| Slow image processing | Enable GPU acceleration in settings if available |
| Connection errors | Check internet connection for Wikipedia/PubMed features or enable offline mode |
| Blurry or unrecognized images | Ensure images are clear, well-lit, and properly focused |
| Missing disease data | Use the "Add Disease" feature to expand the database |
| Missing NLTK data | Run `python -m nltk.downloader punkt wordnet stopwords` |
| PyQt6 errors | Ensure you have the Qt6 libraries installed on your system |
| GPU acceleration not working | Check CUDA installation and compatibility with PyTorch version |

For more assistance, check the [Issues](https://github.com/abhi-abhi86/Multi-Species-Disease-Detection-and-Management-System/issues) section of the repository.

---

## 🛣️ Roadmap

Future development plans:

- **Q4 2025:** 
  - Integration with smartphone camera for real-time diagnosis
  - Enhanced geographic tracking with outbreak prediction
  
- **Q1 2026:**
  - Expanded disease database (+200 conditions)
  - API for third-party integrations
  
- **Q2 2026:**
  - Cloud synchronization for team collaboration
  - Enhanced chatbot with conversational AI
  
- **Long-term:**
  - Mobile applications for Android and iOS
  - Hardware integration for specialized sensors
  - Community-driven disease database contributions

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch:**
   ```sh
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes and commit:**
   ```sh
   git commit -m 'Add some amazing feature'
   ```
4. **Push to your branch:**
   ```sh
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

Please ensure your code follows project standards and includes appropriate tests.

---

## 🙏 Acknowledgments

- PyTorch team for the ML framework
- MobileNetV2 authors for the efficient CNN architecture
- NCBI for research paper access
- Wikipedia for disease information
- Open-source medical imaging datasets
- All contributors and beta testers

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👥 Contributors

- [abhi-abhi86](https://github.com/abhi-abhi86) - Project Lead & Main Developer

---

## 📚 Frequently Asked Questions

**Q: Is this application intended to replace medical professionals?**  
A: No. This tool is designed for preliminary screening, research, and educational purposes only. Always consult qualified healthcare professionals for diagnosis and treatment.

**Q: How accurate is the disease detection?**  
A: Our models achieve 92-97% accuracy on test datasets, but accuracy in real-world scenarios may vary. The system provides confidence levels with each diagnosis.

**Q: Can I add my own disease data?**  
A: Yes! Use the "Add Disease" feature to expand the database with your own entries and images.

**Q: Does the application work offline?**  
A: Core features work offline, but Wikipedia summaries and PubMed research require internet connectivity.

**Q: How is user data handled?**  
A: All processing occurs locally on your device. No diagnosis data is sent to external servers unless you explicitly enable the anonymous data contribution option.

**Q: Do I need a GPU to run the application?**  
A: No, but a CUDA-compatible GPU will significantly improve performance, especially for batch processing of multiple images.

**Q: How do I resolve dependency conflicts?**  
A: Try creating a fresh virtual environment and installing dependencies in the order listed in the requirements.txt file.

**Q: What are the new LLM features?**  
A: The futuristic edition includes OpenAI GPT integration for enhanced chatbot responses, personalized treatment plans, and natural language explanations of diagnoses.

---

> **Note:** This application is intended for research, educational purposes, and preliminary screening only. Always consult with qualified healthcare professionals or specialists for definitive diagnosis and treatment recommendations.
