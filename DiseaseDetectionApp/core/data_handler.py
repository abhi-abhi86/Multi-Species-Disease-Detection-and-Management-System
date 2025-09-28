import os
import json
import re

# The main directory where individual disease JSON files are stored.
DISEASES_DIR = os.path.join(os.path.dirname(__file__), '..', 'diseases')
# Fallback/supplementary database file.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_database.json')

def load_database():
    """
    Dynamically loads all disease information from the 'diseases' directory and
    supplements it with any unique entries from the legacy 'disease_database.json'.
    This makes the loading process more robust.
    """
    database = []
    loaded_names = set() # Keep track of loaded disease names to avoid duplicates

    # 1. Prioritize loading from the modular 'diseases' directory
    print(f"Searching for disease files in: {DISEASES_DIR}")
    if os.path.exists(DISEASES_DIR):
        for root, _, files in os.walk(DISEASES_DIR):
            for file in files:
                if file.endswith('.json') and not file.startswith('_'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            disease_info = json.load(f)
                            if disease_info and 'name' in disease_info:
                                if disease_info['name'] not in loaded_names:
                                    database.append(disease_info)
                                    loaded_names.add(disease_info['name'])
                            else:
                                print(f"Warning: Skipped empty or invalid JSON: {file_path}")
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Error reading or parsing '{file_path}': {e}")

    # 2. Load from the legacy database file and add any entries that are not already loaded
    if os.path.exists(DB_PATH):
        print(f"Loading supplementary diseases from: {DB_PATH}")
        try:
            with open(DB_PATH, 'r', encoding='utf-8') as f:
                legacy_database = json.load(f)
                for disease_info in legacy_database:
                    if disease_info and 'name' in disease_info:
                        if disease_info['name'] not in loaded_names:
                            database.append(disease_info)
                            loaded_names.add(disease_info['name'])
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading or parsing legacy database '{DB_PATH}': {e}")
    
    if not database:
        print("Warning: No disease files were loaded. The application may not function as expected.")

    print(f"Successfully loaded {len(database)} unique disease entries from all sources.")
    return database

def save_disease(disease_data):
    """
    Saves a new disease as a separate JSON file in the appropriate domain subdirectory.
    This keeps the data organized and easy to manage.
    """
    domain = disease_data.get("domain", "general").lower()
    
    # Create a safe filename from the disease name
    disease_name = disease_data.get("name", "unnamed_disease")
    safe_filename = re.sub(r'[^a-z0-9_]', '', disease_name.lower().replace(' ', '_')) + ".json"
    
    if not safe_filename or safe_filename == ".json":
        safe_filename = "new_disease.json"

    # Determine the directory path and create it if it doesn't exist
    domain_dir = os.path.join(DISEASES_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    file_path = os.path.join(domain_dir, safe_filename)
    
    print(f"Saving new disease to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Save with indentation for readability
            json.dump(disease_data, f, indent=4, ensure_ascii=False)
        return True, None
    except Exception as e:
        print(f"Error saving disease file: {e}")
        return False, str(e)
