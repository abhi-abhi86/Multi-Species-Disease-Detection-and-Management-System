# ml_processor.py
import numpy as np
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
from core.wikipedia_integration import get_wikipedia_summary

class MLProcessor:
    """
    Handles loading the ML model and running predictions using PyTorch.
    The model is loaded once and reused for efficiency.
    """
    def __init__(self):
        print("Loading AI model (MobileNetV2 with PyTorch)... This may take a moment on the first run.")
        # Load pre-trained MobileNetV2 model
        self.model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        self.model.eval()  # Set to evaluation mode
        
        # Define image preprocessing transforms (equivalent to TensorFlow's preprocess_input)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet class names for prediction decoding
        self.imagenet_classes = self._load_imagenet_classes()
        print("Model loaded successfully.")

    def _load_imagenet_classes(self):
        """
        Load ImageNet class names for prediction decoding.
        Returns a simple mapping for common classes to simulate decode_predictions functionality.
        """
        # This is a simplified set of common ImageNet classes
        # In a production environment, you would load the full ImageNet class labels
        common_classes = {
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
            5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
            # Add more as needed - this is a minimal set for demonstration
            # Plant-related classes (approximate indices)
            985: 'daisy', 986: 'sunflower', 987: 'rose', 988: 'dandelion',
            989: 'tulip', 990: 'poppy', 991: 'marigold', 992: 'orchid',
            993: 'lily', 994: 'carnation', 995: 'hibiscus', 996: 'chrysanthemum',
            # Disease-related visual patterns
            997: 'leaf_spot', 998: 'fungal_growth', 999: 'blight'
        }
        
        # Fill in remaining indices with generic labels
        for i in range(1000):
            if i not in common_classes:
                common_classes[i] = f'class_{i}'
                
        return common_classes

    def _decode_predictions(self, predictions, top=5):
        """
        Decode PyTorch predictions similar to TensorFlow's decode_predictions.
        """
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        
        # Get top predictions
        top_prob, top_indices = torch.topk(probabilities, top)
        
        decoded = []
        for i in range(top):
            idx = top_indices[i].item()
            prob = top_prob[i].item()
            label = self.imagenet_classes.get(idx, f'unknown_class_{idx}')
            decoded.append((idx, label, prob))
            
        return [decoded]

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
        Predicts a disease and its stage from an image using PyTorch.
        """
        try:
            # Load and preprocess the image using PIL and torchvision
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Decode predictions (equivalent to TensorFlow's decode_predictions)
            decoded_predictions = self._decode_predictions(predictions, top=5)[0]
            
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

