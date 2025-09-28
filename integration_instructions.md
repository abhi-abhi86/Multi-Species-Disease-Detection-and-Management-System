# Disease Detection Pipeline Integration Instructions

This document provides comprehensive instructions for using the real image-to-disease detection pipeline with the Multi-Species Disease Detection and Management System.

## Overview

The pipeline consists of two main scripts that enable actual AI-based disease detection:

1. **`train_disease_classifier.py`** - Trains a MobileNetV2 classifier on your disease dataset
2. **`predict_disease.py`** - Performs inference on images using the trained model

This replaces the ImageNet label matching with clinically meaningful disease classification.

## Prerequisites

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Additional dependencies for training visualizations:
```bash
pip install matplotlib
```

## Dataset Preparation

### 1. Organize Your Dataset

Structure your disease images in a folder-per-class format:

```
disease_dataset/
├── tomato_late_blight/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── rose_black_spot/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── canine_dermatitis/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── human_eczema/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### 2. Image Quality Guidelines

For best results, ensure your images:
- Are high-resolution (minimum 224x224 pixels)
- Show clear disease symptoms
- Have good lighting and contrast
- Are in standard formats (JPG, PNG, BMP, TIFF, WEBP)
- Represent various stages and angles of the disease

### 3. Dataset Size Recommendations

- **Minimum**: 50 images per disease class
- **Recommended**: 200+ images per disease class
- **Ideal**: 500+ images per disease class

More data generally leads to better model performance.

## Training Your Disease Classifier

### Basic Training

Train a model with default settings:

```bash
python train_disease_classifier.py --dataset_path ./disease_dataset --output_dir ./models
```

### Advanced Training Options

```bash
python train_disease_classifier.py \
    --dataset_path ./disease_dataset \
    --output_dir ./models \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --num_workers 2
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_path` | Required | Path to your organized disease dataset |
| `--output_dir` | `./models` | Directory to save trained models |
| `--batch_size` | `32` | Training batch size (reduce if GPU memory issues) |
| `--epochs` | `25` | Number of training epochs |
| `--learning_rate` | `0.001` | Learning rate for optimizer |
| `--pretrained` | `True` | Use ImageNet pretrained weights |
| `--num_workers` | `4` | Number of data loading workers |

### Training Outputs

The training script generates:
- `disease_classifier_YYYYMMDD_HHMMSS.pth` - Trained model file
- `class_mapping_YYYYMMDD_HHMMSS.json` - Class names and indices
- `training_history_YYYYMMDD_HHMMSS.png` - Training progress visualization

## Using the Trained Model for Prediction

### Single Image Prediction

```bash
python predict_disease.py \
    --model_path ./models/disease_classifier_20240101_120000.pth \
    --image_path ./test_image.jpg
```

### Batch Prediction (Multiple Images)

```bash
python predict_disease.py \
    --model_path ./models/disease_classifier_20240101_120000.pth \
    --image_dir ./test_images/
```

### JSON Output (for Integration)

For programmatic integration, use JSON output:

```bash
python predict_disease.py \
    --model_path ./models/disease_classifier_20240101_120000.pth \
    --image_path ./test_image.jpg \
    --json_output
```

### Prediction Options

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to trained model (.pth file) |
| `--image_path` | Single image for prediction |
| `--image_dir` | Directory of images for batch prediction |
| `--top_k` | Number of top predictions to show (default: 5) |
| `--json_output` | Output in JSON format for integration |
| `--save_results` | Save results to JSON file |
| `--show_all` | Show all top-k predictions |

## Integrating with the Existing Application

### Option 1: Replace the MLProcessor (Recommended)

Modify `DiseaseDetectionApp/core/ml_processor.py` to use your trained model:

```python
# Add at the top of ml_processor.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from predict_disease import DiseasePredictor

class MLProcessor:
    """Enhanced ML processor with custom disease classification."""
    
    def __init__(self, custom_model_path=None):
        # Keep existing ImageNet functionality as fallback
        print("Loading AI model (MobileNetV2 with PyTorch)...")
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.labels = get_imagenet_labels()
        
        # Load custom disease classifier if available
        self.disease_predictor = None
        if custom_model_path and os.path.exists(custom_model_path):
            try:
                self.disease_predictor = DiseasePredictor(custom_model_path)
                print(f"Custom disease classifier loaded: {len(self.disease_predictor.class_names)} classes")
            except Exception as e:
                print(f"Failed to load custom model: {e}")
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if self.labels: 
            print("Model loaded successfully.")
        else: 
            print("Warning: Model loaded, but failed to get ImageNet labels.")

    def predict_from_image(self, image_path, domain, database):
        """Enhanced prediction with custom disease classifier."""
        
        # Try custom disease classifier first
        if self.disease_predictor:
            try:
                predictions = self.disease_predictor.predict_single(image_path, top_k=3)
                
                if predictions and predictions[0][1] > 0.3:  # Confidence threshold
                    # Find matching disease in database
                    predicted_class = predictions[0][0]
                    confidence = predictions[0][1] * 100
                    
                    # Search for disease in database by name similarity
                    domain_diseases = [d for d in database if d.get("domain", "").lower() == domain.lower()]
                    
                    best_match = None
                    highest_score = 0
                    
                    for disease in domain_diseases:
                        disease_name = disease.get('name', '').lower()
                        # Simple name matching - you can improve this
                        if predicted_class.lower().replace('_', ' ') in disease_name or \
                           any(word in disease_name for word in predicted_class.lower().split('_')):
                            score = len([word for word in predicted_class.lower().split('_') 
                                       if word in disease_name]) / len(predicted_class.split('_'))
                            if score > highest_score:
                                highest_score = score
                                best_match = disease
                    
                    if best_match:
                        wiki_summary = get_wikipedia_summary(best_match['name'])
                        predicted_stage = "AI Classification"
                        return best_match, confidence, wiki_summary, predicted_stage
                    else:
                        # Create a generic response for the AI prediction
                        generic_disease = {
                            'name': predicted_class.replace('_', ' ').title(),
                            'description': f'AI-detected disease: {predicted_class.replace("_", " ").title()}',
                            'domain': domain,
                            'causes': ['Unknown - detected by AI classifier'],
                            'prevention': ['Consult with specialists for proper diagnosis'],
                            'solutions': ['Seek professional advice for treatment']
                        }
                        return generic_disease, confidence, f"AI detected: {predicted_class}", "AI Classification"
            except Exception as e:
                print(f"Custom classifier failed: {e}")
        
        # Fallback to original ImageNet-based matching
        return self._original_predict_from_image(image_path, domain, database)
```

### Option 2: Standalone Integration

Use the prediction script as a subprocess in your application:

```python
import subprocess
import json

def predict_disease_external(model_path, image_path):
    """Use external prediction script."""
    try:
        result = subprocess.run([
            'python', 'predict_disease.py',
            '--model_path', model_path,
            '--image_path', image_path,
            '--json_output'
        ], capture_output=True, text=True, check=True)
        
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': str(e)}
```

### Option 3: Create a Hybrid Model Directory

Create a `models` directory in your application:

```
DiseaseDetectionApp/
├── models/
│   ├── plant_diseases.pth
│   ├── human_diseases.pth
│   ├── animal_diseases.pth
│   └── class_mappings/
│       ├── plant_classes.json
│       ├── human_classes.json
│       └── animal_classes.json
├── core/
├── ui/
└── ...
```

Then modify the main application to load domain-specific models.

## Model Performance Tips

### Improving Training Results

1. **Data Quality**:
   - Remove blurry or mislabeled images
   - Ensure balanced classes (similar number of images per disease)
   - Include various lighting conditions and angles

2. **Hyperparameter Tuning**:
   - Reduce learning rate for fine-tuning: `--learning_rate 0.0001`
   - Increase epochs for complex datasets: `--epochs 50`
   - Adjust batch size based on GPU memory

3. **Data Augmentation**:
   - The training script includes built-in augmentation
   - Consider external augmentation tools for more variety

### Monitoring Training

- Watch the training history plot for overfitting
- Validation accuracy should continue improving
- If validation loss increases while training loss decreases, stop training earlier

### Model Selection

- Use the model with the highest validation accuracy
- Test accuracy should be close to validation accuracy
- If there's a large gap, you may have overfitting

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size: `--batch_size 8`
   - Reduce number of workers: `--num_workers 1`

2. **Low accuracy**:
   - Check data quality and labels
   - Ensure sufficient data per class
   - Try different learning rates
   - Increase training epochs

3. **Model not loading**:
   - Verify the .pth file path is correct
   - Check if the model was saved completely
   - Ensure compatible PyTorch versions

### Performance Optimization

1. **For GPU training**:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **For CPU inference**:
   - Models automatically use CPU if CUDA is unavailable
   - Inference is still fast for single images

## Example Workflows

### Complete Pipeline Example

```bash
# 1. Prepare your dataset
mkdir disease_dataset
# ... organize your images by class ...

# 2. Train the model
python train_disease_classifier.py \
    --dataset_path ./disease_dataset \
    --output_dir ./models \
    --epochs 30 \
    --batch_size 16

# 3. Test the model
python predict_disease.py \
    --model_path ./models/disease_classifier_20240101_120000.pth \
    --image_path ./test_image.jpg \
    --show_all

# 4. Integrate with your application
# (Follow integration instructions above)
```

### Domain-Specific Training

For better results, train separate models for each domain:

```bash
# Train plant disease model
python train_disease_classifier.py \
    --dataset_path ./plant_diseases \
    --output_dir ./models/plants

# Train human disease model  
python train_disease_classifier.py \
    --dataset_path ./human_diseases \
    --output_dir ./models/humans

# Train animal disease model
python train_disease_classifier.py \
    --dataset_path ./animal_diseases \
    --output_dir ./models/animals
```

## Next Steps

1. **Collect and curate your disease dataset**
2. **Train your first model with the provided scripts**  
3. **Evaluate the model performance**
4. **Integrate with the existing application**
5. **Iterate and improve based on results**

## Support and Contributions

- Report issues with the training or prediction scripts
- Share your trained models and datasets (if appropriate)
- Contribute improvements to the pipeline

This pipeline transforms the application from ImageNet label matching to actual clinical disease detection, making it significantly more useful for real-world applications.