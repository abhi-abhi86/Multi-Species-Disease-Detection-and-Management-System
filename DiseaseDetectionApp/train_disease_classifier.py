import os
import json
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# ========== CONFIG ==========
DATA_DIR = 'dataset'
MODEL_PATH = 'disease_model.pt'
CLASS_MAP_PATH = 'class_to_name.json'
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
IMG_SIZE = 224
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Save mapping from class index to disease name
class_to_name = {i: name for name, i in train_data.class_to_idx.items()}
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(class_to_name, f)

# Model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Training on", device)
for epoch in range(EPOCHS):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss_sum/total:.4f} - Acc: {correct/total:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print(f"Class mapping saved to {CLASS_MAP_PATH}")
