# DiseaseDetectionApp/train_disease_classifier.py
import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import shutil
import re

# ========== CONFIGURATION ==========
# Make paths relative to the script's location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISEASES_DIR = os.path.join(BASE_DIR, 'diseases')  # Source of the images
USER_ADDED_DISEASES_DIR = os.path.join(BASE_DIR, 'user_added_diseases')  # Additional source
DATA_DIR = os.path.join(BASE_DIR, 'dataset')  # Temporary directory for training
MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.pt')
CLASS_MAP_PATH = os.path.join(BASE_DIR, 'class_to_name.json')

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224


# ====================================

def prepare_dataset_for_training():
    """
    Automates creating a temporary 'dataset' directory from the 'diseases'
    directory, structured correctly for PyTorch's ImageFolder.
    """
    print("--- Preparing Dataset for Training ---")

    if not os.path.exists(DISEASES_DIR):
        print(f"FATAL ERROR: The source 'diseases' directory was not found at '{DISEASES_DIR}'.")
        return False

    if os.path.exists(DATA_DIR):
        print(f"Removing old dataset directory: '{DATA_DIR}'")
        shutil.rmtree(DATA_DIR)
    print(f"Creating new dataset directory: '{DATA_DIR}'")
    os.makedirs(DATA_DIR)

    image_count = 0
    class_count = 0
    # Include diseases/
    for root, _, files in os.walk(DISEASES_DIR):
        if os.path.basename(root) == 'images':
            class_name = os.path.basename(os.path.dirname(root))
            safe_class_name = re.sub(r'[\s/\\:*?"<>|]+', '_', class_name).lower()

            if not safe_class_name:
                continue

            class_dest_dir = os.path.join(DATA_DIR, safe_class_name)
            if not os.path.exists(class_dest_dir):
                os.makedirs(class_dest_dir)
                class_count += 1

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(class_dest_dir, file)
                    shutil.copy(source_path, dest_path)
                    image_count += 1

    # Include user_added_diseases/
    if os.path.exists(USER_ADDED_DISEASES_DIR):
        for root, _, files in os.walk(USER_ADDED_DISEASES_DIR):
            if os.path.basename(root) == 'images':
                class_name = os.path.basename(os.path.dirname(root))
                safe_class_name = re.sub(r'[\s/\\:*?"<>|]+', '_', class_name).lower()

                if not safe_class_name:
                    continue

                class_dest_dir = os.path.join(DATA_DIR, safe_class_name)
                if not os.path.exists(class_dest_dir):
                    os.makedirs(class_dest_dir)
                    class_count += 1

                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        source_path = os.path.join(root, file)
                        dest_path = os.path.join(class_dest_dir, file)
                        shutil.copy(source_path, dest_path)
                        image_count += 1

    if image_count == 0:
        print("FATAL ERROR: No images found to train. Please check the 'diseases' directory structure.")
        return False

    print(f"Successfully prepared {image_count} images for {class_count} classes.")
    print("--- Dataset Ready ---")
    return True


def train_model():
    """
    Prepares the dataset and then trains a MobileNetV2 model, saving the
    trained model and class-to-name mapping file.
    """
    # 1. Prepare the dataset from the 'diseases' folder.
    if not prepare_dataset_for_training():
        return  # Stop if preparation fails

    print("\n--- Starting AI Model Training ---")

    # 2. Set up the device (use MPS GPU if available on macOS, otherwise CUDA GPU, otherwise CPU).
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training will be performed on: {device.type.upper()}")

    # 3. Define data transformations and create data loaders.
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
            print("FATAL ERROR: No class subdirectories found in the prepared dataset folder.")
            return
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Found {len(train_data.classes)} classes: {train_data.classes}")

    # 4. Save the mapping from class index to disease name.
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(idx_to_class, f)
    print(f"Class mapping saved to '{CLASS_MAP_PATH}'")

    # 5. Set up the model architecture.
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

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

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # 8. Save the trained model's weights.
    torch.save(model.state_dict(), MODEL_PATH)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Model state dictionary saved to '{MODEL_PATH}'")


if __name__ == "__main__":
    train_model()
