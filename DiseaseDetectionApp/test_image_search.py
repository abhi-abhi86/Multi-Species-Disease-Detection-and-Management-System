#!/usr/bin/env python3
"""
Test script for the updated ImageFetchWorker to verify Wikipedia prioritization and Google fallback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, QObject, pyqtSignal
from ui.image_search_dialog import ImageFetchWorker

# Initialize QApplication for PyQt5
app = QApplication(sys.argv)

class TestWorker(QObject):
    def __init__(self, disease_name):
        super().__init__()
        self.disease_name = disease_name
        self.result = None
        self.error_msg = None

    def run_test(self):
        thread = QThread()
        worker = ImageFetchWorker(self.disease_name)
        worker.moveToThread(thread)

        worker.finished.connect(self.on_finished)
        worker.error.connect(self.on_error)

        thread.started.connect(worker.run)
        thread.start()
        thread.wait()  # Wait for completion

        return self.result, self.error_msg

    def on_finished(self, data):
        self.result = "Image data received (length: {})".format(len(data))

    def on_error(self, msg):
        self.error_msg = msg

def test_disease(disease_name):
    print(f"Testing disease: {disease_name}")
    tester = TestWorker(disease_name)
    result, error = tester.run_test()
    if result:
        print(f"  Success: {result}")
    elif error:
        print(f"  Error: {error}")
    else:
        print("  No result or error")
    print()

if __name__ == "__main__":
    # Test cases
    test_disease("Lumpy Skin Disease")  # Should have Wikipedia image
    test_disease("Nonexistent Disease")  # Should fallback to Google or error
    test_disease("Ringworm")  # Ambiguous, might disambiguate
    test_disease("")  # Invalid
