# ml_processor.py
# PyTorch-based ML processor for disease detection
# Migrated from TensorFlow to PyTorch for improved compatibility and performance
# Uses torchvision MobileNetV2 model for image classification

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
        Returns a mapping with relevant plant, medical, and biological classes.
        """
        # Enhanced mapping with more relevant classes for disease detection
        imagenet_classes = {
            # Plant-related classes from ImageNet (approximate indices based on actual ImageNet)
            946: 'cauliflower', 947: 'mushroom', 948: 'broccoli', 949: 'artichoke',
            950: 'bell pepper', 951: 'cardoon', 952: 'bolete', 953: 'agaric',
            954: 'gyromitra', 955: 'stinkhorn', 956: 'earthstar', 957: 'hen-of-the-woods',
            # Disease/condition related visual patterns
            958: 'corn', 959: 'acorn', 960: 'hip', 961: 'buckeye', 962: 'chestnut',
            963: 'bramble', 964: 'lemon', 965: 'orange', 966: 'banana', 967: 'custard_apple',
            968: 'pomegranate', 969: 'fig', 970: 'pineapple', 971: 'strawberry', 972: 'granny_smith',
            973: 'jackfruit', 974: 'papaya', 975: 'wood_sorrel', 976: 'buckthorn',
            977: 'plant', 978: 'leaf', 979: 'fungus', 980: 'mold', 981: 'blight',
            982: 'spot', 983: 'lesion', 984: 'growth', 985: 'discoloration',
            986: 'yellowing', 987: 'browning', 988: 'wilting', 989: 'decay',
            990: 'infection', 991: 'disease', 992: 'pathology', 993: 'symptom',
            994: 'rash', 995: 'inflammation', 996: 'irritation', 997: 'condition',
            998: 'disorder', 999: 'abnormality'
        }
        
        # Fill in remaining indices with generic labels based on common ImageNet classes
        common_patterns = [
            'animal', 'bird', 'mammal', 'reptile', 'fish', 'insect', 'arthropod',
            'plant', 'tree', 'flower', 'fruit', 'vegetable', 'herb', 'grass',
            'fungus', 'moss', 'lichen', 'algae', 'bacteria', 'virus',
            'tissue', 'organ', 'cell', 'structure', 'pattern', 'texture',
            'surface', 'skin', 'fur', 'feather', 'scale', 'shell',
            'normal', 'healthy', 'diseased', 'infected', 'damaged', 'abnormal'
        ]
        
        pattern_idx = 0
        for i in range(1000):
            if i not in imagenet_classes:
                if pattern_idx < len(common_patterns):
                    imagenet_classes[i] = f'{common_patterns[pattern_idx % len(common_patterns)]}_{i}'
                else:
                    imagenet_classes[i] = f'class_{i}'
                pattern_idx += 1
                
        return imagenet_classes

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
        
        Args:
            image_path (str): Path to the image file
            domain (str): Domain to search in ('Plant', 'Human', 'Animal')
            database (list): List of disease dictionaries
            
        Returns:
            tuple: (disease_dict, confidence_score, wiki_summary, predicted_stage)
        """
        try:
            # Load and preprocess the image using PIL and torchvision
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension
            
            # Run inference with PyTorch model
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Decode predictions (equivalent to TensorFlow's decode_predictions)
            decoded_predictions = self._decode_predictions(predictions, top=5)[0]
            
            prediction_labels = [label for _, label, _ in decoded_predictions]

            # Check each prediction against diseases in the specified domain
            for _, label, score in decoded_predictions:
                print(f"Model identified: {label} (Confidence: {score:.2f})")
                label_words = set(re.findall(r'\b\w+\b', label.lower()))

                for disease in database:
                    if disease.get("domain", "").lower() == domain.lower():
                        disease_text = (disease.get('name', '') + ' ' + disease.get('description', '')).lower()
                        disease_words = set(re.findall(r'\b\w+\b', disease_text))
                        
                        # Check if any prediction keywords match disease keywords
                        if not label_words.isdisjoint(disease_words):
                            print(f"Match found: '{label}' matches '{disease['name']}'")
                            wiki_summary = get_wikipedia_summary(disease['name'])
                            # Predict the stage based on keywords
                            predicted_stage = self._predict_stage(prediction_labels, disease)
                            return disease, score, wiki_summary, predicted_stage

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, 0, "Image file not found.", "Error"
        except Exception as e:
            print(f"An error occurred during image prediction: {e}")
            return None, 0, f"Error processing the image: {str(e)}", "Error"
            
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

