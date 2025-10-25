                                          
                              
                                                            
                                                                      
                                                                  

WATERMARK_AUTHOR = "abhi-abhi86"
WATERMARK_CHECK = True

def check_watermark():
    """Check if watermark is intact. Application will fail if removed."""
    if not WATERMARK_CHECK or WATERMARK_AUTHOR != "abhi-abhi86":
        print("ERROR: Watermark protection violated. Application cannot start.")
        print("Made by: abhi-abhi86")
        import sys
        sys.exit(1)

                         
check_watermark()

import torch
import os
import json
import re
from PIL import Image
from torchvision import models, transforms
from .wikipedia_integration import get_wikipedia_summary
from .google_search import search_google_for_summary

try:
    from fuzzywuzzy import process
except ImportError:
    process = None

                                 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'disease_model.pt')
CLASS_MAP_PATH = os.path.join(BASE_DIR, '..', 'class_to_name.json')
IMG_SIZE = 224

                                                 
IMAGE_CONFIDENCE_THRESHOLD = 0.50
SYMPTOM_CONFIDENCE_THRESHOLD_STRONG = 75
SYMPTOM_CONFIDENCE_THRESHOLD_WEAK = 60
HEALTHY_CLASS_NAMES = ('healthy', 'normal', 'clear_skin')


def get_custom_labels():
    if not os.path.exists(CLASS_MAP_PATH):
        print(f"FATAL ERROR: Class mapping file not found at '{CLASS_MAP_PATH}'.")
        print("Please run 'train_disease_classifier.py' to create it.")
        return None
    try:
        with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
            class_to_name = json.load(f)
        return {int(k): v for k, v in class_to_name.items()}
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading or parsing class mapping file: {e}")
        return None


class MLProcessor:
    def __init__(self):
        print("Initializing custom AI model processor...")
        self.labels = get_custom_labels()
        self.model = None
        self.num_classes = 0

        if not self.labels:
            print("Could not initialize MLProcessor because class labels are missing.")
            return

        self.num_classes = len(self.labels)
        print(f"Found {self.num_classes} classes: {list(self.labels.values())}")

        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(self.model.last_channel, self.num_classes)

        if not os.path.exists(MODEL_PATH):
            print(f"FATAL ERROR: Custom model file not found at '{MODEL_PATH}'.")
            self.model = None
            return

        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            self.model.eval()
            print("Custom disease detection model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model state dictionary: {e}")
            self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

                                                                   
        self.prediction_cache = {}

    def predict_from_image(self, image_path, domain, database):
        if not self.model:
            return None, 0, "The AI model is not loaded.", "Error"

                           
        cache_key = f"{image_path}_{domain}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                logits = self.model(batch_t)
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                top_prob, top_idx = torch.max(probabilities, 0)

            primary_confidence = top_prob.item()
            predicted_class_name = self.labels.get(top_idx.item(), "Unknown")

            if predicted_class_name.lower() in HEALTHY_CLASS_NAMES:
                healthy_result = {
                    'name': 'Healthy',
                    'description': 'The model predicts the sample is healthy.',
                    'solution': 'No action needed.',
                }
                return healthy_result, primary_confidence * 100, "N/A", "Healthy"

            if primary_confidence < IMAGE_CONFIDENCE_THRESHOLD:
                                             
                uncertain_result = {
                    'name': 'No Confident Match Found',
                    'description': 'The AI model could not identify a known disease with high confidence.',
                    'solution': 'Please consult a professional for a definitive diagnosis.',
                }
                return uncertain_result, primary_confidence * 100, "N/A", "Uncertain"

                                    
                                                                                             
            best_match_disease = None
            predicted_name_lower = predicted_class_name.lower()

                                                                             
            for disease_data in database:
                if disease_data.get("domain", "").lower() == domain.lower():
                    db_internal_id = disease_data.get("internal_id", "")
                    if db_internal_id.lower() == predicted_name_lower:
                        best_match_disease = disease_data
                        break

                                                                        
            if not best_match_disease:
                for disease_data in database:
                    if disease_data.get("domain", "").lower() == domain.lower():
                        db_name = disease_data.get("name", "")
                        sanitized_db_name = re.sub(r'[\s/\\:*?"<>|]+', '_', db_name).lower()
                        if sanitized_db_name == predicted_name_lower:
                            best_match_disease = disease_data
                            break

                                                                        
            if not best_match_disease:
                print(f"No exact match for '{predicted_name_lower}'. Trying a contains-match fallback.")
                for disease_data in database:
                    if disease_data.get("domain", "").lower() == domain.lower():
                        db_name = disease_data.get("name", "")
                        sanitized_db_name = re.sub(r'[\s/\\:*?"<>|]+', '_', db_name).lower()
                                                                                
                        if predicted_name_lower in sanitized_db_name or sanitized_db_name in predicted_name_lower:
                            print(
                                f"Found a partial match: AI predicted '{predicted_name_lower}', matched with DB entry '{sanitized_db_name}'")
                            best_match_disease = disease_data
                            break
                                

            if best_match_disease:
                                                                                                               
                wiki_summary = get_wikipedia_summary(best_match_disease['name'])
                return best_match_disease, primary_confidence * 100, wiki_summary, "Detected from Image"
            else:
                                                                                          
                                                                                        
                predicted_clean = predicted_class_name.replace('_', ' ')
                for other_domain in ["Plant", "Human", "Animal"]:
                    if other_domain.lower() != domain.lower():
                        for disease_data in database:
                            if disease_data.get("domain", "").lower() == other_domain.lower():
                                db_name = disease_data.get("name", "")
                                sanitized_db_name = re.sub(r'[\s/\\:*?"<>|]+', '_', db_name).lower()
                                if sanitized_db_name == predicted_class_name.lower():
                                    suggestion_msg = f"This image appears to be of a {other_domain.lower()} disease. Please switch to the '{other_domain}' tab for accurate diagnosis."
                                    return {
                                        'name': f"Wrong Domain - {predicted_clean.title()}",
                                        'description': suggestion_msg,
                                        'solution': f"Switch to the {other_domain} tab and try again.",
                                        'stages': {},
                                        'causes': '',
                                        'risk_factors': '',
                                        'preventive_measures': '',
                                        'image_url': disease_data.get('image_url', '')
                                    }, primary_confidence * 100, "", f"Suggested: Switch to {other_domain} Tab"
                                                                                  
                print(f"No exact match for '{predicted_class_name}'. Attempting fuzzy matching fallback.")
                from fuzzywuzzy import process
                disease_names = [d.get("name", "") for d in database if d.get("domain", "").lower() == domain.lower()]
                if disease_names:
                    best_match, score = process.extractOne(predicted_clean, disease_names)
                    if score >= 60:                             
                        print(f"Fuzzy match found: '{predicted_class_name}' -> '{best_match}' (score: {score})")
                        for disease_data in database:
                            if disease_data.get("domain", "").lower() == domain.lower() and disease_data.get("name", "") == best_match:
                                wiki_summary = get_wikipedia_summary(disease_data['name'])
                                return disease_data, primary_confidence * 100, wiki_summary, "Detected from Image (Fuzzy Match)"
                                                          
                print(
                    f"AI prediction '{predicted_class_name}' not in local DB for domain '{domain}'. Attempting web search...")
                google_summary = search_google_for_summary(predicted_clean)
                wiki_summary = get_wikipedia_summary(predicted_clean)

                if google_summary and "Could not perform search" not in google_summary:
                    web_result = {
                        'name': predicted_class_name.replace('_', ' ').title(),
                        'description': google_summary,
                        'solution': 'No local data available. Please consult online resources or a professional.',
                    }
                    result = (web_result, primary_confidence * 100, wiki_summary, "Found via Web Search")
                else:
                    error_msg = f"AI's prediction '{predicted_class_name}' was not found in the local database for the '{domain}' domain, and the online search feature is either not configured or failed."
                    result = (None, 0, error_msg, "Database Mismatch")

                              
            self.prediction_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"Unexpected error during image prediction: {e}")
            result = (None, 0, f"An error occurred while processing the image: {e}", "Processing Error")
                                                                
            self.prediction_cache[cache_key] = result
            return result


def predict_from_symptoms(symptoms, domain, database):
    """Predicts a disease from symptoms with graded confidence levels."""
    if process is None:
        return None, 0, "The 'fuzzywuzzy' library is not installed.", "Library Missing"

    domain_candidates = {d['name']: d for d in database if d.get("domain", "").lower() == domain.lower()}
    if not domain_candidates:
        return None, 0, f"No diseases found for the '{domain}' domain.", "Not applicable"

                                                                  
    disease_names = list(domain_candidates.keys())

                     
    results = process.extract(symptoms.lower(), disease_names, limit=3)

    if not results or results[0][1] < SYMPTOM_CONFIDENCE_THRESHOLD_WEAK:
        return None, 0, "Could not find a strong match for the specified symptoms.", "Not applicable"

    best_match_name, primary_confidence = results[0]
    best_match_disease = domain_candidates[best_match_name]
    wiki_summary = get_wikipedia_summary(best_match_disease['name'])
    predicted_stage = "Uncertain (Possible Match)" if primary_confidence < SYMPTOM_CONFIDENCE_THRESHOLD_STRONG else "Inferred from Symptoms"

    return best_match_disease, primary_confidence, wiki_summary, predicted_stage

