                      
"""
Comprehensive test script for disease detection across all species (plant, human, animal).
Tests image analysis, symptom analysis, and report generation for each disease type.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from core.ml_processor import MLProcessor
from core.data_handler import load_database
from core.worker import DiagnosisWorker
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, QObject, pyqtSignal

                                   
app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()

class DiagnosisTester(QObject):
    """Test class for running diagnosis tests in a controlled manner."""

    def __init__(self, domain, disease_name=None, symptoms=None, image_path=None):
        super().__init__()
        self.domain = domain
        self.disease_name = disease_name
        self.symptoms = symptoms
        self.image_path = image_path
        self.result = None
        self.confidence = 0.0
        self.wiki_summary = ""
        self.stage = ""
        self.pubmed_summary = ""
        self.error_msg = None

    def run_test(self):
        """Run the diagnosis test and return results."""
        try:
                                            
            database = load_database()
            ml_processor = MLProcessor()

                                  
            thread = QThread()
            worker = DiagnosisWorker(ml_processor, self.image_path, self.symptoms, self.domain, database)
            worker.moveToThread(thread)

                             
            worker.finished.connect(self.on_finished)
            worker.error.connect(self.on_error)

                          
            thread.started.connect(worker.run)
            thread.start()
            thread.wait(10000)                         

                      
            if thread.isRunning():
                worker.stop()
                thread.quit()
                thread.wait()

            return {
                'success': self.result is not None,
                'result': self.result,
                'confidence': self.confidence,
                'wiki_summary': self.wiki_summary,
                'stage': self.stage,
                'pubmed_summary': self.pubmed_summary,
                'error': self.error_msg
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def on_finished(self, result, confidence, wiki, stage, pubmed, domain):
        self.result = result
        self.confidence = confidence
        self.wiki_summary = wiki
        self.stage = stage
        self.pubmed_summary = pubmed

    def on_error(self, error_msg, domain):
        self.error_msg = error_msg

def test_plant_diseases():
    """Test plant disease detection."""
    print("🌱 TESTING PLANT DISEASES")
    print("=" * 50)

    plant_diseases = [
        "powdery mildew",
        "citrus canker",
        "rose black spot",
        "areca nut"
    ]

    for disease in plant_diseases:
        print(f"\nTesting Plant Disease: {disease}")
        print("-" * 30)

                            
        symptoms = f"I have a plant with {disease} symptoms"
        tester = DiagnosisTester("plant", symptoms=symptoms)
        result = tester.run_test()

        if result['success']:
            print(f"✅ Diagnosis: {result['result']['name']}")
            print(".2f")
            print(f"📋 Stage: {result['stage']}")
            print(f"📖 Summary: {result['wiki_summary'][:100]}...")
        else:
            print(f"❌ Error: {result['error']}")

def test_human_diseases():
    """Test human disease detection."""
    print("\n👤 TESTING HUMAN DISEASES")
    print("=" * 50)

    human_diseases = [
        "acne vulgaris",
        "eczema",
        "smoker's lung",
        "aids"
    ]

    for disease in human_diseases:
        print(f"\nTesting Human Disease: {disease}")
        print("-" * 30)

                            
        symptoms = f"I have symptoms of {disease}"
        tester = DiagnosisTester("human", symptoms=symptoms)
        result = tester.run_test()

        if result['success']:
            print(f"✅ Diagnosis: {result['result']['name']}")
            print(".2f")
            print(f"📋 Stage: {result['stage']}")
            print(f"📖 Summary: {result['wiki_summary'][:100]}...")
        else:
            print(f"❌ Error: {result['error']}")

def test_animal_diseases():
    """Test animal disease detection."""
    print("\n🐾 TESTING ANIMAL DISEASES")
    print("=" * 50)

    animal_diseases = [
        "lumpy skin disease",
        "sarcoptic mange",
        "swine erysipelas"
    ]

    for disease in animal_diseases:
        print(f"\nTesting Animal Disease: {disease}")
        print("-" * 30)

                            
        symptoms = f"My animal has {disease} symptoms"
        tester = DiagnosisTester("animal", symptoms=symptoms)
        result = tester.run_test()

        if result['success']:
            print(f"✅ Diagnosis: {result['result']['name']}")
            print(".2f")
            print(f"📋 Stage: {result['stage']}")
            print(f"📖 Summary: {result['wiki_summary'][:100]}...")
        else:
            print(f"❌ Error: {result['error']}")

def test_image_analysis():
    """Test image-based disease detection."""
    print("\n🖼️  TESTING IMAGE ANALYSIS")
    print("=" * 50)

                                          
    image_dir = "sample_images"
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, file)
                print(f"\nTesting Image: {file}")
                print("-" * 30)

                                                                   
                domain = "plant"                    
                if "human" in file.lower():
                    domain = "human"
                elif "animal" in file.lower():
                    domain = "animal"

                tester = DiagnosisTester(domain, image_path=image_path)
                result = tester.run_test()

                if result['success']:
                    print(f"✅ Diagnosis: {result['result']['name']}")
                    print(".2f")
                    print(f"📋 Stage: {result['stage']}")
                else:
                    print(f"❌ Error: {result['error']}")
    else:
        print("No sample_images directory found. Skipping image analysis tests.")

def main():
    """Run all disease detection tests."""
    print("🧬 MULTI-SPECIES DISEASE DETECTION TEST SUITE")
    print("=" * 60)
    print("Testing disease detection across plants, humans, and animals...")
    print()

    try:
                                        
        database = load_database()
        print(f"📚 Database loaded: {len(database)} diseases")
        domains = {}
        for disease in database:
            domain = disease.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1

        print("📊 Diseases by domain:")
        for domain, count in domains.items():
            print(f"   {domain.capitalize()}: {count} diseases")
        print()

                   
        test_plant_diseases()
        test_human_diseases()
        test_animal_diseases()
        test_image_analysis()

        print("\n🎉 ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
