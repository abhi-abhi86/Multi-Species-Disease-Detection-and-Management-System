import os
import json
import re

# The main directory where individual disease JSON files are stored.
DISEASES_DIR = os.path.join(os.path.dirname(__file__), '..', 'diseases')
# Fallback database file, now primarily used for legacy purposes or if the directory is empty.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_database.json')

def load_database():
    """
    Dynamically loads all disease information from JSON files within the 'diseases' directory.
    This approach is more modular and scalable than a single database file.
    """
    database = []
    print(f"Searching for disease files in: {DISEASES_DIR}")

    if not os.path.exists(DISEASES_DIR):
        print(f"Warning: Diseases directory not found at '{DISEASES_DIR}'.")
        # Fallback to the old database file if the new directory structure doesn't exist.
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []

    # Walk through all subdirectories (e.g., 'plant', 'human', 'animal')
    for root, _, files in os.walk(DISEASES_DIR):
        for file in files:
            # Process only JSON files, and ignore templates or empty files.
            if file.endswith('.json') and not file.startswith('_'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        disease_info = json.load(f)
                        # Basic validation to ensure the file is not empty
                        if disease_info:
                            database.append(disease_info)
                        else:
                            print(f"Warning: Skipped empty or invalid JSON file: {file_path}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from '{file_path}': {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading '{file_path}': {e}")
    
    if not database:
        print("Warning: No disease files were loaded. The application may not function as expected.")

    print(f"Successfully loaded {len(database)} disease entries.")
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
    
    if not safe_filename:
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
