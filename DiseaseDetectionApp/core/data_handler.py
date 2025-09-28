import os
import json
import re

# The main directory where individual disease JSON files are stored.
DISEASES_DIR = os.path.join(os.path.dirname(__file__), '..', 'diseases')
# Fallback/supplementary database file.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'disease_database.json')
# Directory where generic disease images are stored.
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'image')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')

def find_disease_images_in_disease_folder(disease_json_path, disease_name):
    """
    Look for image files in the same directory (and subdirectories) as the disease JSON file.
    Returns a list of image paths (relative to project root).
    """
    images = []
    safe_name = re.sub(r'[^a-z0-9_]', '', disease_name.lower().replace(' ', '_'))
    base_dir = os.path.dirname(disease_json_path)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                # Match by file name (optionally by disease name)
                if safe_name in file.lower():
                    rel_path = os.path.relpath(os.path.join(root, file), os.path.join(os.path.dirname(__file__), '..'))
                    images.append(rel_path)
    return images

def find_disease_images(disease_name, domain=None):
    """
    Finds image files related to a disease in the generic image directory.
    Looks for images named after the disease or inside a domain subdirectory.
    Returns a list of image file paths (relative to IMAGES_DIR).
    """
    images = []
    safe_name = re.sub(r'[^a-z0-9_]', '', disease_name.lower().replace(' ', '_'))
    domain = domain.lower() if domain else None

    search_dirs = []
    if domain:
        search_dirs.append(os.path.join(IMAGES_DIR, domain))
    search_dirs.append(IMAGES_DIR)

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file in os.listdir(search_dir):
                if file.lower().startswith(safe_name) and file.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(os.path.relpath(os.path.join(search_dir, file), IMAGES_DIR))
    # Return paths relative to project root for consistency
    return [os.path.join('image', img) for img in images]

def load_database():
    """
    Loads all disease information from the 'diseases' directory, including images found
    in both the disease folder and the generic 'image' folder, and supplements with 'data' if needed.
    """
    database = []
    loaded_names = set() # To avoid duplicates

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
                                disease_name = disease_info['name']
                                domain = disease_info.get('domain', None)
                                if disease_name not in loaded_names:
                                    # Attach images from the disease folder
                                    images = find_disease_images_in_disease_folder(file_path, disease_name)
                                    # Also attach images from the generic images folder
                                    images += find_disease_images(disease_name, domain)
                                    if images:
                                        disease_info["images"] = images
                                    database.append(disease_info)
                                    loaded_names.add(disease_name)
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
                        disease_name = disease_info['name']
                        domain = disease_info.get('domain', None)
                        if disease_name not in loaded_names:
                            # Only attach images from the generic images folder for legacy data
                            images = find_disease_images(disease_name, domain)
                            if images:
                                disease_info["images"] = images
                            database.append(disease_info)
                            loaded_names.add(disease_name)
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
    disease_name = disease_data.get("name", "unnamed_disease")
    safe_filename = re.sub(r'[^a-z0-9_]', '', disease_name.lower().replace(' ', '_')) + ".json"
    
    if not safe_filename or safe_filename == ".json":
        safe_filename = "new_disease.json"

    domain_dir = os.path.join(DISEASES_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    file_path = os.path.join(domain_dir, safe_filename)
    
    print(f"Saving new disease to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(disease_data, f, indent=4, ensure_ascii=False)
        return True, None
    except Exception as e:
        print(f"Error saving disease file: {e}")
        return False, str(e)
