                      
"""
Main entry point for the Multi-Species Disease Detection and Management System.
This script allows running the application from the project root directory.
"""

import sys
import os

                                                          
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiseaseDetectionApp'))

                                     
from DiseaseDetectionApp.main import main

if __name__ == "__main__":
    main()
