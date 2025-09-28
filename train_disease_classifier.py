#!/usr/bin/env python3
"""
Disease Classifier Training Script

This script trains a MobileNetV2 classifier on a folder-per-class dataset structure.
The trained model and class mappings are saved for use with predict_disease.py.

Expected dataset structure:
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...

Usage:
python train_disease_classifier.py --dataset_path /path/to/dataset --output_dir ./models
"""

import os
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import matplotlib.pyplot as plt


class DiseaseDataset(Dataset):
    """Custom dataset for disease images organized in folders by class."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subdirectories for each class
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
    
    def _make_dataset(self):
        """Create list of (image_path, class_index) tuples."""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if self._is_valid_image(img_path):
                    samples.append((img_path, class_idx))
        return samples
    
    def _is_valid_image(self, img_path):
        """Check if file is a valid image."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return os.path.splitext(img_path.lower())[1] in valid_extensions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                image = Image.new('RGB', (224, 224), color='black')
            return image, class_idx


def get_transforms(is_training=True):
    """Get data transforms for training and validation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_model(num_classes, pretrained=True):
    """Create MobileNetV2 model for disease classification."""
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v2(weights=None)
    
    # Modify the classifier for our number of classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    return model


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot([acc.cpu().numpy() for acc in train_accs], label='Train Accuracy', color='blue')
    ax2.plot([acc.cpu().numpy() for acc in val_accs], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train disease classifier')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory with class subdirectories')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained model and class mapping')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ImageNet weights')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = DiseaseDataset(
        root_dir=args.dataset_path,
        transform=get_transforms(is_training=True)
    )
    
    print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes:")
    for i, class_name in enumerate(full_dataset.classes):
        count = sum(1 for _, class_idx in full_dataset.samples if class_idx == i)
        print(f"  {class_name}: {count} images")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    # Apply different transforms to validation and test sets
    val_dataset.dataset.transform = get_transforms(is_training=False)
    test_dataset.dataset.transform = get_transforms(is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create model
    model = create_model(num_classes=len(full_dataset.classes), pretrained=args.pretrained)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  → New best validation accuracy: {best_val_acc:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"disease_classifier_{timestamp}.pth"
    mapping_filename = f"class_mapping_{timestamp}.json"
    history_filename = f"training_history_{timestamp}.png"
    
    model_path = os.path.join(args.output_dir, model_filename)
    mapping_path = os.path.join(args.output_dir, mapping_filename)
    history_path = os.path.join(args.output_dir, history_filename)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'num_classes': len(full_dataset.classes),
        'class_names': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'training_args': vars(args),
        'timestamp': timestamp
    }, model_path)
    
    # Save class mapping
    class_mapping = {
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': {idx: class_name for class_name, idx in full_dataset.class_to_idx.items()},
        'num_classes': len(full_dataset.classes),
        'model_file': model_filename,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'timestamp': timestamp,
        'training_args': vars(args)
    }
    
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Save training history plot
    plot_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Class mapping saved to: {mapping_path}")
    print(f"Training history plot saved to: {history_path}")
    
    print(f"\nTo use this model for prediction, run:")
    print(f"python predict_disease.py --model_path {model_path} --image_path /path/to/image.jpg")


if __name__ == '__main__':
    main()