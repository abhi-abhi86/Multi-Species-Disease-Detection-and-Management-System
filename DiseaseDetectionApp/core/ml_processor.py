# ml_processor.py
import numpy as np
import re
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from core.wikipedia_integration import get_wikipedia_summary

class MLProcessor:
    """
    Handles loading the ML model and running predictions.
    The model is loaded once and reused for efficiency.
    """
    def __init__(self):
        print("Loading AI model (MobileNetV2)... This may take a moment on the first run.")
        # Load the pre-trained MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully.")

    def predict_from_image(self, image_path, domain, database):
        """
        Predicts a disease from an image using a pre-trained deep learning model.
        It identifies objects in the image and matches them to diseases in the database.
        """
        try:
            # 1. Load and preprocess the image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)

            # 2. Get predictions from the model
            predictions = self.model.predict(img_preprocessed)
            decoded_predictions = decode_predictions(predictions, top=5)[0]

            # 3. Match predictions to the disease database
            # The model predicts general classes (e.g., 'tomato', 'dog'). We need to find a disease
            # in our database that matches these predictions.
            for _, label, score in decoded_predictions:
                print(f"Model identified: {label} (Confidence: {score:.2f})")
                label_words = set(re.findall(r'\b\w+\b', label.lower()))

                # Search for a disease that matches the predicted label
                for disease in database:
                    if disease.get("domain", "").lower() == domain.lower():
                        disease_text = (disease.get('name', '') + ' ' + disease.get('description', '')).lower()
                        disease_words = set(re.findall(r'\b\w+\b', disease_text))
                        
                        # If a keyword from the prediction matches a word in the disease info
                        if not label_words.isdisjoint(disease_words):
                            print(f"Match found: '{label}' matches '{disease['name']}'")
                            wiki_summary = get_wikipedia_summary(disease['name'])
                            return disease, score, wiki_summary

        except Exception as e:
            print(f"An error occurred during image prediction: {e}")
            return None, 0, "Error processing the image."
            
        return None, 0, ""

def predict_from_symptoms(symptoms, domain, database):
    """
    Predicts from symptoms using keyword matching and fetches a Wikipedia summary.
    (This function remains unchanged but is kept for consistency).
    """
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if not candidates:
        return None, 0, ""

    user_symptoms = set(re.findall(r'\b\w+\b', symptoms.lower()))
    if not user_symptoms:
        return None, 0, ""

    best_match = None
    max_score = 0

    for disease in candidates:
        disease_text = (
            disease.get('name', '') + ' ' +
            disease.get('description', '') + ' ' +
            ' '.join(disease.get('stages', {}).values())
        ).lower()
        disease_words = set(re.findall(r'\b\w+\b', disease_text))
        score = len(user_symptoms.intersection(disease_words))
        
        if score > max_score:
            max_score = score
            best_match = disease

    if best_match:
        # Normalize confidence against the number of user-provided symptoms
        confidence = min(1.0, max_score / len(user_symptoms)) if user_symptoms else 0
        wiki_summary = get_wikipedia_summary(best_match['name'])
        return best_match, confidence, wiki_summary
    
    return None, 0, ""

