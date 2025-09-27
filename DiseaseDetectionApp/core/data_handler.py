import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_database.json')

def load_database():
    """
    Loads the disease database from the JSON file.
    Now includes UTF-8 encoding and error handling for corrupted files.
    """
    if not os.path.exists(DB_PATH):
        # Create the directory and an empty database file if they don't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []
    
    try:
        # Read the file with explicit UTF-8 encoding
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load disease database. It might be corrupted or missing. Error: {e}")
        # Return an empty list to allow the application to continue running
        return []

def save_disease(disease_data):
    """
    Saves new disease data to the JSON file with UTF-8 encoding.
    """
    db = load_database()
    db.append(disease_data)
    # Write the file with explicit UTF-8 encoding to ensure consistency
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
