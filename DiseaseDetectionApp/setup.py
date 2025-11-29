"""
Setup script for creating a standalone macOS application using py2app
"""
from setuptools import setup

APP = ['main.py']
DATA_FILES = [
    ('diseases', ['diseases/human/acne vulgaris/Acne Vulgaris.json',
                  'diseases/human/Aids/package.json',
                  'diseases/human/eczema/eczema.json',
                  "diseases/human/smoker's lung/smoker's lung.json",
                  'diseases/plant/areca nut/areca_nut.json',
                  'diseases/plant/citrus canker/citrus_canker.json',
                  'diseases/plant/powdery mildew/powdery_mildew.json',
                  'diseases/plant/rose black spot/rose_black_spot.json',
                  'diseases/animal/lumpy skin disease/lumpy_skin_disease.json',
                  'diseases/animal/sarcoptic mange/sarcoptic_mange.json',
                  'diseases/animal/swine erysipelas/swine_erysipelas.json']),
    ('', ['disease_model.pt', 'class_to_name.json', '.developer_info.json'])
]
OPTIONS = {
    'argv_emulation': True,
    'packages': ['PySide6', 'torch', 'torchvision', 'PIL', 'requests', 'bs4'],
    'includes': ['core.ml_processor', 'core.worker', 'core.data_handler',
                 'core.wikipedia_integration', 'core.ncbi_integration',
                 'core.google_search', 'ui.main_window', 'ui.create_spinner',
                 'ui.add_disease_dialog', 'core.retraining_worker'],
    'excludes': ['tkinter', 'unittest', 'pdb', 'pydoc'],
    'iconfile': None,
    'plist': {
        'CFBundleName': 'Multi-Species Disease Detection',
        'CFBundleDisplayName': 'Multi-Species Disease Detection',
        'CFBundleGetInfoString': "AI-powered disease detection system",
        'CFBundleIdentifier': "com.abhishek.diseasedetection",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
        'NSHumanReadableCopyright': "© 2025 Abhishek MG"
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
