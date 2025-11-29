#!/usr/bin/env python3
"""
Dependency Verification Script
Tests all required imports for the Multi-Species Disease Detection System
"""

import sys

def test_imports():
    """Test all required package imports"""
    results = []
    
    # Test each dependency
    dependencies = [
        ("PyQt5", "from PyQt5.QtWidgets import QApplication"),
        ("PyQt5.QtCore", "from PyQt5.QtCore import Qt, pyqtSignal, QThread"),
        ("PyQt5.QtGui", "from PyQt5.QtGui import QPixmap, QFont"),
        ("torch", "import torch"),
        ("torchvision", "from torchvision import models, transforms"),
        ("PIL", "from PIL import Image"),
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("sklearn", "from sklearn import metrics"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("requests", "import requests"),
        ("bs4", "from bs4 import BeautifulSoup"),
        ("wikipedia", "import wikipedia"),
        ("geopy", "from geopy.geocoders import Nominatim"),
        ("folium", "import folium"),
        ("fuzzywuzzy", "from fuzzywuzzy import process"),
        ("Levenshtein", "import Levenshtein"),
        ("rank_bm25", "from rank_bm25 import BM25Okapi"),
        ("reportlab", "from reportlab.platypus import SimpleDocTemplate"),
        ("transformers", "from transformers import pipeline"),
        ("openai", "import openai"),
        ("json", "import json"),
        ("os", "import os"),
        ("re", "import re"),
    ]
    
    print("=" * 70)
    print("DEPENDENCY VERIFICATION TEST")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for name, import_stmt in dependencies:
        try:
            exec(import_stmt)
            status = "✓ PASS"
            passed += 1
            color = "\033[92m"  # Green
        except ImportError as e:
            status = f"✗ FAIL: {str(e)}"
            failed += 1
            color = "\033[91m"  # Red
        except Exception as e:
            status = f"✗ ERROR: {str(e)}"
            failed += 1
            color = "\033[91m"  # Red
        
        reset = "\033[0m"
        print(f"{color}{status:12}{reset} {name:20} ({import_stmt})")
    
    print()
    print("=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(dependencies)} total")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All dependencies are correctly installed!")
        return 0
    else:
        print(f"\n✗ {failed} dependencies failed. Please install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
