#!/usr/bin/env python3
"""
Disease Prediction Script

This script loads a trained disease classifier model and performs inference on images.
It outputs the predicted disease class with confidence scores.

Usage:
python predict_disease.py --model_path ./models/disease_classifier_20240101_120000.pth --image_path /path/to/image.jpg

For batch prediction:
python predict_disease.py --model_path ./models/disease_classifier_20240101_120000.pth --image_dir /path/to/images/

For integration with existing application:
python predict_disease.py --model_path ./models/disease_classifier_20240101_120000.pth --image_path /path/to/image.jpg --json_output
"""

import os
import argparse
import json
import time
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np


class DiseasePredictor:
    """Class for loading and using trained disease classification models."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the disease predictor.
        
        Args:
            model_path: Path to the saved model file
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.idx_to_class = {}
        self.metadata = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the trained model and metadata."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract metadata
        self.metadata = {
            'num_classes': checkpoint.get('num_classes'),
            'class_names': checkpoint.get('class_names', []),
            'class_to_idx': checkpoint.get('class_to_idx', {}),
            'best_val_acc': checkpoint.get('best_val_acc'),
            'test_acc': checkpoint.get('test_acc'),
            'timestamp': checkpoint.get('timestamp'),
            'training_args': checkpoint.get('training_args', {})
        }
        
        self.class_names = self.metadata['class_names']
        self.idx_to_class = {idx: class_name for class_name, idx in self.metadata['class_to_idx'].items()}
        
        # Create model architecture
        self.model = mobilenet_v2(weights=None)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.last_channel, 
            self.metadata['num_classes']
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")
        print(f"Validation accuracy: {self.metadata.get('best_val_acc', 'Unknown'):.4f}")
        print(f"Test accuracy: {self.metadata.get('test_acc', 'Unknown'):.4f}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
    
    def predict_single(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict disease for a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, confidence) tuples, sorted by confidence (descending)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.class_names)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = self.idx_to_class[idx.item()]
                confidence = prob.item()
                predictions.append((class_name, confidence))
            
            return predictions
    
    def predict_batch(self, image_paths: List[str], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict diseases for multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return per image
            
        Returns:
            Dictionary mapping image paths to prediction lists
        """
        results = {}
        
        for image_path in image_paths:
            try:
                predictions = self.predict_single(image_path, top_k)
                results[image_path] = predictions
            except Exception as e:
                print(f"Error predicting for {image_path}: {e}")
                results[image_path] = [("Error", 0.0)]
        
        return results
    
    def predict_directory(self, image_dir: str, top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Predict diseases for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            top_k: Number of top predictions to return per image
            
        Returns:
            Dictionary mapping image paths to prediction lists
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all valid image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for filename in os.listdir(image_dir):
            if os.path.splitext(filename.lower())[1] in valid_extensions:
                image_paths.append(os.path.join(image_dir, filename))
        
        if not image_paths:
            print(f"No valid images found in {image_dir}")
            return {}
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        return self.predict_batch(image_paths, top_k)


def format_predictions(predictions: List[Tuple[str, float]], show_all: bool = False) -> str:
    """Format predictions for display."""
    if not predictions:
        return "No predictions available"
    
    lines = []
    for i, (class_name, confidence) in enumerate(predictions):
        if i == 0:  # Top prediction
            lines.append(f"🎯 Primary Prediction: {class_name} ({confidence:.2%} confidence)")
        elif show_all:
            lines.append(f"   Alternative #{i}: {class_name} ({confidence:.2%})")
        
        if not show_all and i == 0:  # Only show top prediction
            break
    
    return "\n".join(lines)


def save_results_json(results: Dict, output_path: str, metadata: Dict = None):
    """Save prediction results to JSON file."""
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_metadata': metadata or {},
        'predictions': {}
    }
    
    for image_path, predictions in results.items():
        output_data['predictions'][image_path] = [
            {'class': class_name, 'confidence': float(confidence)}
            for class_name, confidence in predictions
        ]
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Predict disease from images using trained classifier')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file (.pth)')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_path', type=str,
                            help='Path to a single image for prediction')
    input_group.add_argument('--image_dir', type=str,
                            help='Directory containing images for batch prediction')
    
    # Output options
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--json_output', action='store_true',
                       help='Output results in JSON format (for integration)')
    parser.add_argument('--save_results', type=str,
                       help='Save results to JSON file (specify output path)')
    parser.add_argument('--show_all', action='store_true',
                       help='Show all top-k predictions (not just the top one)')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = DiseasePredictor(args.model_path)
        
        # Run predictions
        results = {}
        
        if args.image_path:
            # Single image prediction
            print(f"\nAnalyzing image: {args.image_path}")
            predictions = predictor.predict_single(args.image_path, args.top_k)
            results[args.image_path] = predictions
            
        elif args.image_dir:
            # Batch prediction
            print(f"\nAnalyzing images in directory: {args.image_dir}")
            results = predictor.predict_directory(args.image_dir, args.top_k)
        
        # Output results
        if args.json_output:
            # JSON format for integration
            output = {
                'success': True,
                'model_info': {
                    'classes': predictor.class_names,
                    'validation_accuracy': predictor.metadata.get('best_val_acc'),
                    'test_accuracy': predictor.metadata.get('test_acc')
                },
                'predictions': {}
            }
            
            for image_path, predictions in results.items():
                output['predictions'][image_path] = [
                    {'class': class_name, 'confidence': float(confidence)}
                    for class_name, confidence in predictions
                ]
            
            print(json.dumps(output, indent=2))
        
        else:
            # Human-readable format
            print(f"\n{'='*60}")
            print("DISEASE PREDICTION RESULTS")
            print(f"{'='*60}")
            
            for image_path, predictions in results.items():
                print(f"\n📁 Image: {os.path.basename(image_path)}")
                print(f"   Path: {image_path}")
                print(f"   {format_predictions(predictions, args.show_all)}")
                
                if predictions and predictions[0][1] < 0.5:
                    print(f"   ⚠️  Low confidence - consider checking image quality or training data")
        
        # Save results to file if requested
        if args.save_results:
            save_results_json(results, args.save_results, predictor.metadata)
        
        print(f"\n{'='*60}")
        print(f"Model Information:")
        print(f"  Classes: {len(predictor.class_names)}")
        print(f"  Validation Accuracy: {predictor.metadata.get('best_val_acc', 'Unknown'):.4f}")
        print(f"  Test Accuracy: {predictor.metadata.get('test_acc', 'Unknown'):.4f}")
        print(f"  Device: {predictor.device}")
        
    except Exception as e:
        if args.json_output:
            error_output = {
                'success': False,
                'error': str(e),
                'predictions': {}
            }
            print(json.dumps(error_output, indent=2))
        else:
            print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())