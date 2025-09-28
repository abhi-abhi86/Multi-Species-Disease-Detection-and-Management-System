# DiseaseDetectionApp/predict_disease.py
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
import argparse

# --- Constants ---
# Use absolute paths based on this script's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.pt')
CLASS_MAP_PATH = os.path.join(BASE_DIR, 'class_to_name.json')
IMG_SIZE = 224

def predict_image(image_path, model, idx_to_class, transform):
    """
    Takes an image path and a loaded model, and returns the predicted disease name and confidence.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        # Apply transformations
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # Make prediction
        with torch.no_grad():
            logits = model(batch_t)
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Get the top prediction
            top_prob, top_idx = torch.max(probabilities, 0)
            pred_idx = top_idx.item()
            confidence = top_prob.item()
            
            predicted_class = idx_to_class.get(pred_idx, "Unknown")
            
        return predicted_class, confidence
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None, 0
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, 0

def main():
    """
    Main function to run the command-line prediction tool.
    """
    # Set up argument parser for a better command-line interface.
    parser = argparse.ArgumentParser(description="Predict a disease from an image using a trained model.")
    parser.add_argument("image_path", type=str, help="The full path to the image file for diagnosis.")
    args = parser.parse_args()

    # 1. Check for model and class map files.
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
        print("Error: `disease_model.pt` or `class_to_name.json` not found.")
        print("Please run `train_disease_classifier.py` first to train and save the model.")
        return

    # 2. Load class mapping.
    try:
        with open(CLASS_MAP_PATH) as f:
            idx_to_class = json.load(f)
            # Convert string keys from JSON back to integers
            idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    except Exception as e:
        print(f"Error loading class mapping file: {e}")
        return

    # 3. Load the model architecture and weights.
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.last_channel, len(idx_to_class))
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # 4. Define image transformations (must match training).
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Run prediction and print results.
    predicted_disease, confidence = predict_image(args.image_path, model, idx_to_class, transform)
    
    if predicted_disease:
        print("\n--- Prediction Result ---")
        print(f"  Image File: {os.path.basename(args.image_path)}")
        print(f"  Predicted Disease: {predicted_disease}")
        print(f"  Confidence: {confidence:.2%}")
        print("-------------------------\n")

if __name__ == "__main__":
    main()
