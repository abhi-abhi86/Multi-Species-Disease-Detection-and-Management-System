# ml_processor.py
import torch
import torchvision.transforms as transforms
# Updated import for model weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import numpy as np
import re
import json
import os
import requests
from core.wikipedia_integration import get_wikipedia_summary

# Path to the ImageNet class index file
IMAGENET_CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'imagenet_class_index.json')

# A threshold for how similar the prediction must be to a disease description to be considered a match.
# This value (e.g., 0.1 means 10% similarity) can be adjusted to make matching stricter or more lenient.
SIMILARITY_THRESHOLD = 0.1

def get_imagenet_labels():
    """
    Downloads and loads the ImageNet class labels if they don't exist locally.
    """
    if not os.path.exists(IMAGENET_CLASS_INDEX_PATH):
        print("Downloading ImageNet class index...")
        try:
            url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            # Added a timeout to the request
            response = requests.get(url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(IMAGENET_CLASS_INDEX_PATH, 'w') as f:
                f.write(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ImageNet class index: {e}")
            # Create an empty file to avoid repeated download attempts
            with open(IMAGENET_CLASS_INDEX_PATH, 'w') as f:
                json.dump({}, f)
            return None

    try:
        with open(IMAGENET_CLASS_INDEX_PATH) as f:
            class_idx = json.load(f)
        # Creates a dictionary mapping the class index (integer) to the class name (string)
        return {int(k): v[1] for k, v in class_idx.items()}
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error reading or parsing ImageNet class index file: {e}")
        return None


class MLProcessor:
    """
    Handles loading the PyTorch model and running predictions.
    The model is loaded once and reused for efficiency.
    """
    def __init__(self):
        print("Loading AI model (MobileNetV2 with PyTorch)... This may take a moment on the first run.")
        # Load a pre-trained MobileNetV2 model using the recommended weights API
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.eval()  # Set the model to evaluation mode (important for inference)
        
        self.labels = get_imagenet_labels()
        
        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.labels:
            print("Model loaded successfully.")
        else:
            print("Warning: Model loaded, but failed to get ImageNet labels. Predictions may not be interpretable.")

    def _predict_stage(self, prediction_labels, disease_info):
        """
        Analyzes prediction labels to determine the most likely disease stage.
        """
        best_stage = "Undetermined"
        max_score = 0
        
        prediction_keywords = set()
        for label in prediction_labels:
            prediction_keywords.update(re.findall(r'\b\w+\b', label.lower()))

        for stage_name, stage_desc in disease_info.get("stages", {}).items():
            stage_words = set(re.findall(r'\b\w+\b', stage_desc.lower()))
            score = len(prediction_keywords.intersection(stage_words))
            
            if score > max_score:
                max_score = score
                best_stage = stage_name
        
        return best_stage

    def predict_from_image(self, image_path, domain, database):
        """
        Predicts a disease and its stage from an image using PyTorch.
        Refined to use Jaccard similarity for more accurate matching.
        """
        if not self.labels:
            return None, 0, "ImageNet labels are missing or corrupted.", "Error"
            
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad(): # Disable gradient calculation for inference
                out = self.model(batch_t)

            _, indices = torch.sort(out, descending=True)
            percentages = torch.nn.functional.softmax(out, dim=1)[0]
            
            top_predictions = [(self.labels.get(idx.item(), "unknown"), percentages[idx.item()].item()) for idx in indices[0][:5]]
            prediction_labels = [label for label, _ in top_predictions]

            for label, score in top_predictions:
                print(f"Model identified: {label} (Confidence: {score:.2%})")
                label_words = set(re.findall(r'\b\w+\b', label.lower()))

                for disease in database:
                    if disease.get("domain", "").lower() == domain.lower():
                        disease_text = (disease.get('name', '') + ' ' + disease.get('description', '')).lower()
                        disease_words = set(re.findall(r'\b\w+\b', disease_text))
                        
                        # Calculate Jaccard similarity (intersection over union)
                        intersection = len(label_words.intersection(disease_words))
                        union = len(label_words.union(disease_words))
                        similarity = intersection / union if union > 0 else 0

                        # Check if similarity exceeds the defined threshold
                        if similarity > SIMILARITY_THRESHOLD:
                            print(f"Match found with similarity {similarity:.2f}: '{label}' matches '{disease['name']}'")
                            wiki_summary = get_wikipedia_summary(disease['name'])
                            predicted_stage = self._predict_stage(prediction_labels, disease)
                            # Return confidence as a percentage score (0-100)
                            return disease, score * 100, wiki_summary, predicted_stage

        except Exception as e:
            print(f"An error occurred during image prediction: {e}")
            return None, 0, f"Error processing the image: {e}", "Error"
            
        return None, 0, "No matching disease found in the database for the top predictions.", "Not applicable"

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
        # Normalize confidence score based on the number of matching keywords
        confidence = (max_score / len(user_symptoms)) * 100 if user_symptoms else 0
        wiki_summary = get_wikipedia_summary(best_match['name'])
        predicted_stage = "Based on provided symptoms"
        return best_match, min(confidence, 100.0), wiki_summary, predicted_stage # Cap confidence at 100
    
    return None, 0, "", "Not applicable"

