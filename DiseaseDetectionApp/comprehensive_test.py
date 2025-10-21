# DiseaseDetectionApp/comprehensive_test.py
# Comprehensive test suite for all modules and files

import os
import sys
import importlib
import traceback
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ModuleTester:
    def __init__(self):
        self.results = {}
        self.failed_modules = []

    def test_import(self, module_name, description=""):
        """Test if a module can be imported successfully."""
        try:
            module = importlib.import_module(module_name)
            self.results[module_name] = {
                'status': 'PASS',
                'type': 'import',
                'description': description,
                'error': None
            }
            print(f"✓ {module_name}: Import successful")
            return module
        except Exception as e:
            error_msg = f"Import failed: {str(e)}"
            self.results[module_name] = {
                'status': 'FAIL',
                'type': 'import',
                'description': description,
                'error': error_msg
            }
            self.failed_modules.append(module_name)
            print(f"✗ {module_name}: {error_msg}")
            return None

    def test_function(self, module_name, func_name, *args, **kwargs):
        """Test if a function in a module can be called."""
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            result = func(*args, **kwargs)
            self.results[f"{module_name}.{func_name}"] = {
                'status': 'PASS',
                'type': 'function',
                'description': f"Function call test",
                'error': None,
                'result': str(result)[:100] if result is not None else None
            }
            print(f"✓ {module_name}.{func_name}: Function call successful")
            return result
        except Exception as e:
            error_msg = f"Function call failed: {str(e)}"
            self.results[f"{module_name}.{func_name}"] = {
                'status': 'FAIL',
                'type': 'function',
                'description': f"Function call test",
                'error': error_msg
            }
            print(f"✗ {module_name}.{func_name}: {error_msg}")
            return None

    def test_file_existence(self, file_path, description=""):
        """Test if a file exists."""
        full_path = project_root / file_path
        if full_path.exists():
            self.results[file_path] = {
                'status': 'PASS',
                'type': 'file',
                'description': description,
                'error': None
            }
            print(f"✓ {file_path}: File exists")
            return True
        else:
            self.results[file_path] = {
                'status': 'FAIL',
                'type': 'file',
                'description': description,
                'error': 'File not found'
            }
            self.failed_modules.append(file_path)
            print(f"✗ {file_path}: File not found")
            return False

    def test_json_validity(self, file_path):
        """Test if a JSON file is valid."""
        full_path = project_root / file_path
        if not full_path.exists():
            self.results[f"{file_path}_json"] = {
                'status': 'FAIL',
                'type': 'json',
                'description': 'JSON validity test',
                'error': 'File does not exist'
            }
            return False

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.results[f"{file_path}_json"] = {
                'status': 'PASS',
                'type': 'json',
                'description': 'JSON validity test',
                'error': None
            }
            print(f"✓ {file_path}: Valid JSON")
            return True
        except Exception as e:
            error_msg = f"Invalid JSON: {str(e)}"
            self.results[f"{file_path}_json"] = {
                'status': 'FAIL',
                'type': 'json',
                'description': 'JSON validity test',
                'error': error_msg
            }
            print(f"✗ {file_path}: {error_msg}")
            return False

    def run_comprehensive_test(self):
        """Run all tests."""
        print("🧬 COMPREHENSIVE MODULE TEST SUITE")
        print("=" * 50)

        # Test core modules
        print("\n📁 Testing Core Modules:")
        core_modules = [
            ('core.ml_processor', 'ML processing module'),
            ('core.worker', 'Diagnosis worker module'),
            ('core.data_handler', 'Data handling module'),
            ('core.wikipedia_integration', 'Wikipedia integration module'),
            ('core.ncbi_integration', 'NCBI/PubMed integration module'),
            ('core.google_search', 'Google search integration module'),
        ]

        for module_name, description in core_modules:
            self.test_import(module_name, description)

        # Test UI modules
        print("\n🖥️  Testing UI Modules:")
        ui_modules = [
            ('ui.main_window', 'Main window UI module'),
            ('ui.create_spinner', 'Spinner creation UI module'),
        ]

        for module_name, description in ui_modules:
            self.test_import(module_name, description)

        # Test main application files
        print("\n🚀 Testing Main Application Files:")
        main_files = [
            ('main.py', 'Main application entry point'),
            ('predict_disease.py', 'Disease prediction script'),
            ('train_disease_classifier.py', 'Model training script'),
        ]

        for file_name, description in main_files:
            if self.test_file_existence(file_name, description):
                # Try to import if it's a Python file
                if file_name.endswith('.py'):
                    module_name = file_name[:-3]  # Remove .py extension
                    self.test_import(module_name, f"Import test for {description}")

        # Test existing test files
        print("\n🧪 Testing Test Files:")
        test_files = [
            ('test_disease_detection.py', 'Disease detection test suite'),
            ('test_train_model.py', 'Model training test suite'),
        ]

        for file_name, description in test_files:
            if self.test_file_existence(file_name, description):
                module_name = file_name[:-3]
                self.test_import(module_name, f"Import test for {description}")

        # Test critical data files
        print("\n📊 Testing Data Files:")
        data_files = [
            ('disease_model.pt', 'Trained ML model file'),
            ('class_to_name.json', 'Class name mapping file'),
        ]

        for file_name, description in data_files:
            self.test_file_existence(file_name, description)
            if file_name.endswith('.json'):
                self.test_json_validity(file_name)

        # Test disease database structure
        print("\n🗂️  Testing Disease Database:")
        diseases_dir = project_root / 'diseases'
        if diseases_dir.exists():
            json_files = list(diseases_dir.rglob('*.json'))
            print(f"Found {len(json_files)} disease JSON files")
            for json_file in json_files[:5]:  # Test first 5 files
                rel_path = json_file.relative_to(project_root)
                self.test_json_validity(str(rel_path))
        else:
            print("✗ diseases directory not found")

        # Test function calls on key modules
        print("\n⚙️  Testing Key Functions:")
        try:
            # Test data loading
            data_handler = self.test_import('core.data_handler')
            if data_handler:
                self.test_function('core.data_handler', 'load_database')
        except:
            pass

        try:
            # Test ML processor initialization
            ml_proc = self.test_import('core.ml_processor')
            if ml_proc:
                self.test_function('core.ml_processor', 'MLProcessor')
        except:
            pass

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate a comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")

        if self.failed_modules:
            print(f"\n❌ Failed Modules/Components ({len(self.failed_modules)}):")
            for module in self.failed_modules:
                result = self.results.get(module, {})
                print(f"  - {module}: {result.get('error', 'Unknown error')}")

        print("
📄 Detailed Results:"        for name, result in self.results.items():
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"  {status_icon} {name} ({result['type']})")

        # Save detailed results to file
        results_file = project_root / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📁 Detailed results saved to: {results_file}")

        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {failed_tests} TESTS FAILED - Check the detailed results above")

if __name__ == "__main__":
    tester = ModuleTester()
    tester.run_comprehensive_test()
