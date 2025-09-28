# ml_processor.py
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import re
import json
import os
import requests
from core.wikipedia_integration import get_wikipedia_summary

IMAGENET_CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'imagenet_class_index.json')
SIMILARITY_THRESHOLD = 0.1 # Increased threshold slightly as the new logic is more precise

def get_imagenet_labels():
    """Downloads and loads the ImageNet class labels."""
    if not os.path.exists(IMAGENET_CLASS_INDEX_PATH):
        print("Downloading ImageNet class index...")
        try:
            url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(IMAGENET_CLASS_INDEX_PATH, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ImageNet class index: {e}")
            with open(IMAGENET_CLASS_INDEX_PATH, 'w', encoding='utf-8') as f: json.dump({}, f)
            return None

    try:
        with open(IMAGENET_CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
            class_idx = json.load(f)
        return {int(k): v[1] for k, v in class_idx.items()}
    except (json.JSONDecodeError, IndexError, FileNotFoundError) as e:
        print(f"Error reading or parsing ImageNet class index file: {e}")
        return None

class MLProcessor:
    """Handles loading the PyTorch model and running predictions."""
    def __init__(self):
        print("Loading AI model (MobileNetV2 with PyTorch)...")
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.labels = get_imagenet_labels()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.labels: print("Model loaded successfully.")
        else: print("Warning: Model loaded, but failed to get ImageNet labels.")

    def _predict_stage(self, prediction_labels, disease_info):
        """Analyzes prediction labels to determine the most likely disease stage."""
        best_stage, max_score = "Undetermined", 0
        prediction_keywords = {word for label in prediction_labels for word in re.findall(r'\b\w+\b', label.lower())}
        for stage_name, stage_desc in disease_info.get("stages", {}).items():
            stage_words = set(re.findall(r'\b\w+\b', stage_desc.lower()))
            score = len(prediction_keywords.intersection(stage_words))
            if score > max_score:
                max_score, best_stage = score, stage_name
        return best_stage
    
    def _calculate_jaccard_similarity(self, set1, set2):
        """Helper function to calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def predict_from_image(self, image_path, domain, database):
        """
        Predicts a disease using a weighted similarity score that prioritizes matches
        in the disease name over the description.
        """
        if not self.labels:
            return None, 0, "ImageNet labels are missing.", "Error"
            
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                out = self.model(batch_t)

            _, indices = torch.sort(out, descending=True)
            percentages = torch.nn.functional.softmax(out, dim=1)[0]
            
            top_predictions = [(self.labels.get(idx.item(), "unknown"), percentages[idx.item()].item()) for idx in indices[0][:5]]
            prediction_labels = [label for label, _ in top_predictions]

            # --- ENHANCED LOGIC WITH WEIGHTING ---
            best_match_disease = None
            highest_final_score = 0
            
            domain_diseases = [d for d in database if d.get("domain", "").lower() == domain.lower()]

            for label, model_conf in top_predictions:
                print(f"Model identified: {label} (Confidence: {model_conf:.2%})")
                label_words = set(re.findall(r'\b\w+\b', label.lower()))

                for disease in domain_diseases:
                    disease_name_words = set(re.findall(r'\b\w+\b', disease.get('name', '').lower()))
                    disease_desc_words = set(re.findall(r'\b\w+\b', disease.get('description', '').lower()))
                    
                    # Calculate similarity for name and description separately
                    name_similarity = self._calculate_jaccard_similarity(label_words, disease_name_words)
                    desc_similarity = self._calculate_jaccard_similarity(label_words, disease_desc_words)
                    
                    # Apply weights: name match is 2.5x more important than description match
                    weighted_similarity = (0.7 * name_similarity) + (0.3 * desc_similarity)
                    
                    # Combine model confidence with our similarity score for a final score
                    final_score = model_conf * weighted_similarity
                    
                    if final_score > highest_final_score:
                        highest_final_score = final_score
                        best_match_disease = disease

            # After checking all possibilities, see if the best match is good enough
            if best_match_disease and highest_final_score > SIMILARITY_THRESHOLD / 10: # Adjust threshold for final score
                # Convert final score to a percentage for display
                final_confidence_pct = min(highest_final_score * 100, 100.0)
                print(f"Best match found: '{best_match_disease['name']}' with final confidence {final_confidence_pct:.2f}%")
                
                wiki_summary = get_wikipedia_summary(best_match_disease['name'])
                predicted_stage = self._predict_stage(prediction_labels, best_match_disease)
                return best_match_disease, final_confidence_pct, wiki_summary, predicted_stage
            # --- END OF ENHANCED LOGIC ---

        except Exception as e:
            print(f"An error occurred during image prediction: {e}")
            return None, 0, f"Error processing the image: {e}", "Error"
            
        return None, 0, "No matching disease found in the database for the top AI predictions.", "Not applicable"


def predict_from_symptoms(symptoms, domain, database):
    """Predicts from symptoms using keyword matching."""
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if not candidates: return None, 0, "", "Not applicable"

    user_symptoms = set(re.findall(r'\b\w+\b', symptoms.lower()))
    if not user_symptoms: return None, 0, "", "Not applicable"

    best_match, max_score = None, 0
    for disease in candidates:
        disease_text = (
            disease.get('name', '') + ' ' +
            disease.get('description', '') + ' ' +
            ' '.join(disease.get('stages', {}).values())
        ).lower()
        disease_words = set(re.findall(r'\b\w+\b', disease_text))
        score = len(user_symptoms.intersection(disease_words))
        
        if score > max_score:
            max_score, best_match = score, disease

    if best_match:
        confidence = (max_score / len(user_symptoms)) * 100 if user_symptoms else 0
        wiki_summary = get_wikipedia_summary(best_match['name'])
        predicted_stage = "Based on provided symptoms"
        return best_match, min(confidence, 100.0), wiki_summary, predicted_stage
    
    return None, 0, "", "Not applicable"

