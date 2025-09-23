import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_database.json')

def load_database():
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'w') as f:
            json.dump([], f)
        return []
    with open(DB_PATH, 'r') as f:
        return json.load(f)

def save_disease(disease_data):
    db = load_database()
    db.append(disease_data)
    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)