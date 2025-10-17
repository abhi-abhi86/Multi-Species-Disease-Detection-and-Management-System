#!/usr/bin/env python3
"""
Comprehensive test script for all core, data, diseases, and ui modules.
Tests imports, basic functionality, and data validation.
"""

import sys
import os
import json
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test importing all modules."""
    print("🧪 TESTING IMPORTS")
    print("=" * 50)

    modules_to_test = [
        # Core modules
        'core.data_handler',
        'core.ml_processor',
        'core.worker',
        'core.report_generator',
        'core.html_report_generator',
        'core.llm_integrator',
        'core.ncbi_integration',
        'core.wikipedia_integration',
        'core.google_search',
        'core.prepare_dataset',
        # UI modules (import without GUI)
        'ui.main_window',
        'ui.add_disease_dialog',
        'ui.chatbot_dialog',
        'ui.image_search_dialog',
        'ui.create_spinner',
        'ui.map_dialog',
    ]

    failed_imports = []

    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n❌ Failed imports: {len(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_core_functions():
    """Test basic functions in core modules."""
    print("\n🔧 TESTING CORE FUNCTIONS")
    print("=" * 50)

    try:
        from core.data_handler import load_database
        print("Testing data_handler.load_database()...")
        database = load_database()
        if database and len(database) > 0:
            print(f"✅ Loaded {len(database)} diseases")
            # Check structure
            sample = database[0]
            required_keys = ['name', 'domain', 'description']
            if all(key in sample for key in required_keys):
                print("✅ Database structure valid")
            else:
                print("❌ Database structure invalid")
                return False
        else:
            print("❌ Failed to load database")
            return False
    except Exception as e:
        print(f"❌ data_handler test failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.ml_processor import MLProcessor
        print("Testing MLProcessor initialization...")
        ml_proc = MLProcessor()
        if ml_proc.model is None:
            print("⚠️  ML model not loaded (expected if no model file)")
        else:
            print("✅ ML model loaded")
    except Exception as e:
        print(f"❌ MLProcessor test failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.ncbi_integration import get_pubmed_summary
        print("Testing ncbi_integration.get_pubmed_summary()...")
        # Test with a known disease, but limit to avoid long wait
        summary = get_pubmed_summary("test", max_results=1)
        if summary:
            print("✅ PubMed integration working")
        else:
            print("⚠️  PubMed returned empty (may be network issue)")
    except Exception as e:
        print(f"❌ ncbi_integration test failed: {e}")
        return False

    try:
        from core.wikipedia_integration import get_wikipedia_summary
        print("Testing wikipedia_integration.get_wikipedia_summary()...")
        summary = get_wikipedia_summary("Test")
        if summary:
            print("✅ Wikipedia integration working")
        else:
            print("⚠️  Wikipedia returned empty")
    except Exception as e:
        print(f"❌ wikipedia_integration test failed: {e}")
        return False

    print("✅ Core functions tests passed!")
    return True

def test_data_validation():
    """Test data files and diseases structure."""
    print("\n📁 TESTING DATA VALIDATION")
    print("=" * 50)

    # Test disease_database.json if exists
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'disease_database.json')
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                print("✅ disease_database.json valid")
            else:
                print("❌ disease_database.json invalid")
                return False
        except Exception as e:
            print(f"❌ Error reading disease_database.json: {e}")
            return False
    else:
        print("⚠️  disease_database.json not found (using modular diseases/)")

    # Test diseases/ directory
    diseases_dir = os.path.join(os.path.dirname(__file__), 'diseases')
    if os.path.exists(diseases_dir):
        json_files = []
        for root, dirs, files in os.walk(diseases_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        if json_files:
            print(f"Found {len(json_files)} disease JSON files")
            # Test a few
            for i, json_file in enumerate(json_files[:3]):  # Test first 3
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'name' in data and 'domain' in data:
                        print(f"✅ {os.path.basename(json_file)} valid")
                    else:
                        print(f"❌ {os.path.basename(json_file)} missing required fields")
                        return False
                except Exception as e:
                    print(f"❌ Error in {os.path.basename(json_file)}: {e}")
                    return False
        else:
            print("❌ No disease JSON files found")
            return False
    else:
        print("❌ diseases/ directory not found")
        return False

    print("✅ Data validation passed!")
    return True

def test_ui_imports():
    """Test UI module imports (without GUI)."""
    print("\n🖥️  TESTING UI IMPORTS")
    print("=" * 50)

    ui_modules = [
        'ui.main_window',
        'ui.add_disease_dialog',
        'ui.chatbot_dialog',
        'ui.image_search_dialog',
        'ui.create_spinner',
        'ui.map_dialog',
    ]

    failed = []
    for module in ui_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed.append(module)

    if failed:
        print(f"❌ Failed UI imports: {len(failed)}")
        return False
    else:
        print("✅ All UI imports successful!")
        return True

def main():
    """Run all tests."""
    print("🧬 COMPREHENSIVE MODULE TEST SUITE")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Core Functions", test_core_functions()))
    results.append(("Data Validation", test_data_validation()))
    results.append(("UI Imports", test_ui_imports()))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The codebase is functioning correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    return all_passed

if __name__ == "__main__":
    main()
