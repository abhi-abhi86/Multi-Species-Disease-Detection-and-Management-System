# Example: Integration with Existing MLProcessor

This example shows how to modify the existing `MLProcessor` to use the trained disease classifier while maintaining backward compatibility.

## Modified MLProcessor Class

```python
# File: DiseaseDetectionApp/core/ml_processor_enhanced.py
# This is an example of how to modify the existing MLProcessor

import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import re
import json
import requests
from core.wikipedia_integration import get_wikipedia_summary

# Add the parent directory to path to import our predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from predict_disease import DiseasePredictor
    CUSTOM_PREDICTOR_AVAILABLE = True
except ImportError:
    CUSTOM_PREDICTOR_AVAILABLE = False

IMAGENET_CLASS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'imagenet_class_index.json')
CUSTOM_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SIMILARITY_THRESHOLD = 0.1

class EnhancedMLProcessor:
    """Enhanced ML processor with custom disease classification support."""
    
    def __init__(self):
        print("Loading Enhanced AI model (MobileNetV2 with custom classifiers)...")
        
        # Load original ImageNet model as fallback
        self.imagenet_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.imagenet_model.eval()
        self.imagenet_labels = self._get_imagenet_labels()
        
        # Standard transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load custom disease predictors for each domain
        self.custom_predictors = {}
        self._load_custom_predictors()
        
        print("Enhanced model loaded successfully.")
        if self.custom_predictors:
            print(f"Custom predictors loaded for domains: {list(self.custom_predictors.keys())}")
        else:
            print("No custom predictors found. Using ImageNet fallback.")
    
    def _get_imagenet_labels(self):
        """Load ImageNet labels (keeping original functionality)."""
        # ... (keep original implementation)
        try:
            if not os.path.exists(IMAGENET_CLASS_INDEX_PATH):
                # Download logic here...
                pass
            
            with open(IMAGENET_CLASS_INDEX_PATH, 'r', encoding='utf-8') as f:
                class_idx = json.load(f)
            return {int(k): v[1] for k, v in class_idx.items()}
        except:
            return None
    
    def _load_custom_predictors(self):
        """Load custom disease predictors for each domain."""
        if not CUSTOM_PREDICTOR_AVAILABLE or not os.path.exists(CUSTOM_MODELS_DIR):
            return
        
        domain_models = {
            'plant': 'plant_diseases.pth',
            'human': 'human_diseases.pth', 
            'animal': 'animal_diseases.pth'
        }
        
        for domain, model_file in domain_models.items():
            model_path = os.path.join(CUSTOM_MODELS_DIR, model_file)
            if os.path.exists(model_path):
                try:
                    predictor = DiseasePredictor(model_path)
                    self.custom_predictors[domain] = predictor
                    print(f"  ✓ Loaded {domain} disease classifier ({len(predictor.class_names)} classes)")
                except Exception as e:
                    print(f"  ✗ Failed to load {domain} model: {e}")
    
    def predict_from_image(self, image_path, domain, database):
        """
        Enhanced prediction using custom classifiers when available.
        Falls back to ImageNet matching if custom classifier not available or confident.
        """
        domain_key = domain.lower()
        
        # Try custom predictor first
        if domain_key in self.custom_predictors:
            try:
                custom_result = self._predict_with_custom_classifier(
                    image_path, domain_key, database
                )
                if custom_result:
                    return custom_result
            except Exception as e:
                print(f"Custom classifier failed: {e}")
        
        # Fallback to original ImageNet-based prediction
        return self._predict_with_imagenet(image_path, domain, database)
    
    def _predict_with_custom_classifier(self, image_path, domain, database):
        """Use custom disease classifier for prediction."""
        predictor = self.custom_predictors[domain]
        
        # Get predictions from custom model
        predictions = predictor.predict_single(image_path, top_k=3)
        
        if not predictions or predictions[0][1] < 0.3:  # Low confidence threshold
            return None
        
        predicted_class = predictions[0][0]
        confidence = predictions[0][1] * 100
        
        print(f"Custom classifier prediction: {predicted_class} ({confidence:.1f}% confidence)")
        
        # Try to match with database entries
        domain_diseases = [d for d in database if d.get("domain", "").lower() == domain.lower()]
        
        best_match = self._find_database_match(predicted_class, domain_diseases)
        
        if best_match:
            wiki_summary = get_wikipedia_summary(best_match['name'])
            predicted_stage = self._predict_stage([predicted_class], best_match)
            return best_match, confidence, wiki_summary, predicted_stage
        else:
            # Create a new disease entry based on AI prediction
            ai_disease = {
                'name': predicted_class.replace('_', ' ').title(),
                'description': f'AI-detected disease: {predicted_class.replace("_", " ").title()}. Confidence: {confidence:.1f}%',
                'domain': domain.title(),
                'causes': [f'Detected by AI classifier with {confidence:.1f}% confidence'],
                'stages': {'detected': f'AI identified as {predicted_class}'},
                'prevention': ['Consult with domain experts for proper diagnosis and treatment'],
                'solutions': ['Seek professional advice for accurate diagnosis and treatment plans'],
                'is_ai_generated': True
            }
            
            wiki_summary = get_wikipedia_summary(predicted_class.replace('_', ' '))
            return ai_disease, confidence, wiki_summary, "AI Detection"
    
    def _find_database_match(self, predicted_class, domain_diseases):
        """Find the best matching disease in the database."""
        predicted_words = set(predicted_class.lower().replace('_', ' ').split())
        
        best_match = None
        highest_score = 0
        
        for disease in domain_diseases:
            disease_name = disease.get('name', '').lower()
            disease_desc = disease.get('description', '').lower()
            
            # Score based on word overlap
            name_words = set(re.findall(r'\b\w+\b', disease_name))
            desc_words = set(re.findall(r'\b\w+\b', disease_desc))
            
            name_score = len(predicted_words.intersection(name_words)) / len(predicted_words) if predicted_words else 0
            desc_score = len(predicted_words.intersection(desc_words)) / len(predicted_words) if predicted_words else 0
            
            total_score = 0.8 * name_score + 0.2 * desc_score
            
            if total_score > highest_score and total_score > 0.3:  # Minimum similarity threshold
                highest_score = total_score
                best_match = disease
        
        return best_match
    
    def _predict_with_imagenet(self, image_path, domain, database):
        """Original ImageNet-based prediction logic."""
        # ... (keep original implementation from ml_processor.py)
        # This is the fallback when custom classifiers aren't available or confident
        
        if not self.imagenet_labels:
            return None, 0, "ImageNet labels are missing.", "Error"
        
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                out = self.imagenet_model(batch_t)

            _, indices = torch.sort(out, descending=True)
            percentages = torch.nn.functional.softmax(out, dim=1)[0]

            top_predictions = [(self.imagenet_labels.get(idx.item(), "unknown"), 
                              percentages[idx.item()].item()) for idx in indices[0][:5]]
            
            # Continue with original matching logic...
            # (Implementation would continue with the existing similarity matching)
            
            return None, 0, "Using ImageNet fallback - no strong matches found.", "ImageNet Fallback"
            
        except Exception as e:
            return None, 0, f"Error processing image: {e}", "Error"
    
    def _predict_stage(self, prediction_labels, disease_info):
        """Original stage prediction logic."""
        best_stage, max_score = "AI Detected", 0
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
```

## Usage Instructions

1. **Train your models** (one per domain):
   ```bash
   # Train plant disease model
   python train_disease_classifier.py \
       --dataset_path ./plant_disease_data \
       --output_dir ./DiseaseDetectionApp/models \
       --epochs 30

   # Rename the output file
   mv ./DiseaseDetectionApp/models/disease_classifier_*.pth ./DiseaseDetectionApp/models/plant_diseases.pth
   ```

2. **Replace the MLProcessor** in `main_window.py`:
   ```python
   # In DiseaseDetectionApp/ui/main_window.py
   from core.ml_processor_enhanced import EnhancedMLProcessor
   
   # Replace this line:
   # self.ml_processor = MLProcessor()
   # With:
   self.ml_processor = EnhancedMLProcessor()
   ```

3. **Benefits of this integration**:
   - Uses trained disease classifiers when available
   - Falls back to ImageNet matching for backward compatibility
   - Provides confidence scores from actual disease detection
   - Can handle domain-specific models (plants, humans, animals)
   - Maintains existing UI and workflow

## Model Directory Structure

```
DiseaseDetectionApp/
└── models/
    ├── plant_diseases.pth       # Trained plant disease classifier
    ├── human_diseases.pth       # Trained human disease classifier
    ├── animal_diseases.pth      # Trained animal disease classifier
    └── class_mappings/
        ├── plant_classes.json   # Class mappings for plants
        ├── human_classes.json   # Class mappings for humans
        └── animal_classes.json  # Class mappings for animals
```

This integration approach provides:
- **Gradual upgrade path**: Existing functionality remains if no custom models
- **Domain-specific accuracy**: Different models for plants, humans, animals
- **Confidence-based fallback**: Uses ImageNet if custom classifier isn't confident
- **Seamless UI integration**: No changes needed to the user interface