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
        self.model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully.")

    def _predict_stage(self, prediction_labels, disease_info):
        """
        Analyzes prediction labels to determine the most likely disease stage.
        """
        best_stage = "Undetermined"
        max_score = 0
        
        # Get a set of all keywords from the top AI predictions
        prediction_keywords = set()
        for label in prediction_labels:
            prediction_keywords.update(re.findall(r'\b\w+\b', label.lower()))

        # Compare keywords against each stage's description
        for stage_name, stage_desc in disease_info.get("stages", {}).items():
            stage_words = set(re.findall(r'\b\w+\b', stage_desc.lower()))
            score = len(prediction_keywords.intersection(stage_words))
            
            if score > max_score:
                max_score = score
                best_stage = stage_name
        
        return best_stage

    def predict_from_image(self, image_path, domain, database):
        """
        Predicts a disease and its stage from an image.
        """
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            predictions = self.model.predict(img_preprocessed)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            prediction_labels = [label for _, label, _ in decoded_predictions]

            for _, label, score in decoded_predictions:
                print(f"Model identified: {label} (Confidence: {score:.2f})")
                label_words = set(re.findall(r'\b\w+\b', label.lower()))

                for disease in database:
                    if disease.get("domain", "").lower() == domain.lower():
                        disease_text = (disease.get('name', '') + ' ' + disease.get('description', '')).lower()
                        disease_words = set(re.findall(r'\b\w+\b', disease_text))
                        
                        if not label_words.isdisjoint(disease_words):
                            print(f"Match found: '{label}' matches '{disease['name']}'")
                            wiki_summary = get_wikipedia_summary(disease['name'])
                            # Predict the stage based on keywords
                            predicted_stage = self._predict_stage(prediction_labels, disease)
                            return disease, score, wiki_summary, predicted_stage

        except Exception as e:
            print(f"An error occurred during image prediction: {e}")
            return None, 0, "Error processing the image.", "Error"
            
        return None, 0, "", "Not applicable"

def predict_from_symptoms(symptoms, domain, database):
    """
    Predicts from symptoms using keyword matching.
    """
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if not candidates:
        return None, 0, "", "Not applicable"

    user_symptoms = set(re.findall(r'\b\w+\b', symptoms.lower()))
    if not user_symptoms:
        return None, 0, "", "Not applicable"

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
        confidence = min(1.0, max_score / len(user_symptoms)) if user_symptoms else 0
        wiki_summary = get_wikipedia_summary(best_match['name'])
        predicted_stage = "Based on provided symptoms"
        return best_match, confidence, wiki_summary, predicted_stage
    
    return None, 0, "", "Not applicable"

