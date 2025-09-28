# DiseaseDetectionApp/train_disease_classifier.py
import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time

# ========== CONFIGURATION ==========
# Make paths relative to the script's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.pt')
CLASS_MAP_PATH = os.path.join(BASE_DIR, 'class_to_name.json')

# Training parameters
BATCH_SIZE = 16  # Reduced for better compatibility with less powerful hardware
EPOCHS = 20      # Increased for potentially better accuracy
LEARNING_RATE = 0.001
IMG_SIZE = 224
# ====================================

def train_model():
    """
    Trains a MobileNetV2 model on the provided image dataset and saves the
    trained model and class-to-name mapping file.
    """
    print("--- Starting AI Model Training ---")

    # 1. Check if the dataset directory exists.
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"FATAL ERROR: Dataset directory not found or is empty at '{DATA_DIR}'.")
        print("Please create the 'dataset' directory and add subdirectories for each disease with training images.")
        return

    # 2. Set up the device (use GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will be performed on: {device.type.upper()}")

    # 3. Define data transformations and create data loaders.
    # Data augmentation (RandomHorizontalFlip, RandomRotation) helps the model generalize better.
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        train_data = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
        if not train_data.classes:
            print("FATAL ERROR: No class subdirectories found in the dataset folder.")
            return
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Found {len(train_data.classes)} classes: {train_data.classes}")

    # 4. Save the mapping from class index to disease name. This is crucial for predictions.
    # The saved JSON will have keys like "0", "1", etc.
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(idx_to_class, f)
    print(f"Class mapping saved to '{CLASS_MAP_PATH}'")

    # 5. Set up the model architecture.
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze the pre-trained layers. We only want to train the final new layer.
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier layer with a new one tailored to our number of diseases.
    model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))
    model = model.to(device)

    # 6. Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 7. Start the training loop.
    start_time = time.time()
    print("\n--- Training in Progress ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += images.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # 8. Save the trained model's weights.
    torch.save(model.state_dict(), MODEL_PATH)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Model state dictionary saved to '{MODEL_PATH}'")

if __name__ == "__main__":
    train_model()
