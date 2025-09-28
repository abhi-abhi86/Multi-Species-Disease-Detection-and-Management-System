# DiseaseDetectionApp/core/data_handler.py
import os
import json
import re

# --- Constants ---
# Use absolute paths based on this file's location to prevent execution path issues.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DIR = os.path.join(BASE_DIR, '..', 'diseases')
# This legacy file will be used as a fallback or for supplementary data.
LEGACY_DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'disease_database.json')

def load_database():
    """
    Loads all disease information from the modular 'diseases' directory first,
    then supplements it with any unique entries from the legacy 'disease_database.json'.
    This approach is robust and prevents duplicate entries.
    """
    database = []
    loaded_disease_names = set() # Use a set to track loaded diseases and prevent duplicates.

    # 1. Prioritize loading from the modern, modular 'diseases' directory.
    print(f"Searching for disease files in: {DISEASES_DIR}")
    if not os.path.exists(DISEASES_DIR):
        print(f"Warning: The directory '{DISEASES_DIR}' does not exist. No modular diseases will be loaded.")
    else:
        for root, _, files in os.walk(DISEASES_DIR):
            for file in files:
                # Process only non-template JSON files.
                if file.endswith('.json') and not file.startswith('_'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            disease_info = json.load(f)
                            
                            # Basic validation to ensure the JSON is a valid disease entry.
                            if isinstance(disease_info, dict) and 'name' in disease_info:
                                disease_name = disease_info['name'].strip()
                                if disease_name and disease_name.lower() not in loaded_disease_names:
                                    database.append(disease_info)
                                    loaded_disease_names.add(disease_name.lower())
                                else:
                                    print(f"Warning: Skipped duplicate or empty-named disease in '{file_path}'.")
                            else:
                                print(f"Warning: Skipped invalid or non-dictionary JSON file: {file_path}")
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Error reading or parsing '{file_path}': {e}")

    # 2. Load from the legacy database file and add any entries that haven't been loaded yet.
    if os.path.exists(LEGACY_DB_PATH):
        print(f"Loading supplementary diseases from: {LEGACY_DB_PATH}")
        try:
            with open(LEGACY_DB_PATH, 'r', encoding='utf-8') as f:
                legacy_database = json.load(f)
                if isinstance(legacy_database, list):
                    for disease_info in legacy_database:
                        if isinstance(disease_info, dict) and 'name' in disease_info:
                            disease_name = disease_info['name'].strip()
                            if disease_name and disease_name.lower() not in loaded_disease_names:
                                database.append(disease_info)
                                loaded_disease_names.add(disease_name.lower())
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading or parsing legacy database '{LEGACY_DB_PATH}': {e}")
    
    if not database:
        print("CRITICAL WARNING: No disease data was loaded. The application will not be able to provide diagnoses.")
    else:
        print(f"Successfully loaded {len(database)} unique disease entries.")
        
    return database

def save_disease(disease_data):
    """
    Saves a new disease as a clean, separate JSON file in the appropriate
    domain subdirectory inside 'diseases/'.
    """
    try:
        domain = disease_data.get("domain", "general").strip().lower()
        disease_name = disease_data.get("name", "unnamed_disease").strip()
        
        # Create a safe, valid filename from the disease name.
        # "Rose Black Spot" -> "rose_black_spot.json"
        safe_filename = re.sub(r'[\s/\\:*?"<>|]+', '_', disease_name).lower()
        safe_filename = re.sub(r'[^a-z0-9_]', '', safe_filename) + ".json"
        
        if not safe_filename or safe_filename == ".json":
            raise ValueError("Disease name is invalid or empty, cannot create filename.")

        # Create the domain-specific directory if it doesn't exist.
        domain_dir = os.path.join(DISEASES_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        file_path = os.path.join(domain_dir, safe_filename)
        
        print(f"Saving new disease to: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            # `ensure_ascii=False` is good for international characters.
            json.dump(disease_data, f, indent=4, ensure_ascii=False)
        return True, None # Return success status and no error message.
        
    except Exception as e:
        print(f"Error saving disease file: {e}")
        return False, str(e) # Return failure status and the error message.
