##predict_disease.py   
import torch
from torchvision import transforms, models
from PIL import Image
import json

MODEL_PATH = 'disease_model.pt'
CLASS_MAP_PATH = 'class_to_name.json'
IMG_SIZE = 224

# Load class mapping
with open(CLASS_MAP_PATH) as f:
    class_to_name = json.load(f)
    idx_to_class = {int(k): v for k, v in class_to_name.items()}

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(1).item()
        pred_class = idx_to_class[pred_idx]
    return pred_class

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    pred = predict_image(img_path)
    print(f"Predicted Disease: {pred}")
