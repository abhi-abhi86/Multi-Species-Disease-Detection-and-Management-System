# Disease Classifier Training and Prediction Scripts

This directory contains two new scripts that enable real AI-based disease detection for the Multi-Species Disease Detection and Management System:

## Scripts Overview

### 1. `train_disease_classifier.py`
**Purpose**: Train a custom MobileNetV2 classifier on your disease dataset

**Key Features**:
- Supports folder-per-class dataset organization
- Uses transfer learning with ImageNet pretrained weights
- Automatic train/validation/test splitting
- Data augmentation for better generalization
- Training progress visualization
- Model checkpointing (saves best validation model)

**Basic Usage**:
```bash
python train_disease_classifier.py --dataset_path ./my_disease_data --output_dir ./models
```

### 2. `predict_disease.py` 
**Purpose**: Use trained models to classify disease images

**Key Features**:
- Single image or batch prediction
- Confidence scores for predictions
- JSON output for programmatic integration
- Top-k prediction support
- Compatible with models from `train_disease_classifier.py`

**Basic Usage**:
```bash
# Single image
python predict_disease.py --model_path ./models/my_model.pth --image_path ./test_image.jpg

# Batch prediction
python predict_disease.py --model_path ./models/my_model.pth --image_dir ./test_images/

# JSON output (for integration)
python predict_disease.py --model_path ./models/my_model.pth --image_path ./test.jpg --json_output
```

## Quick Start

1. **Prepare your dataset**:
   ```
   my_disease_dataset/
   ├── tomato_blight/
   │   ├── img1.jpg
   │   └── img2.jpg
   ├── rose_black_spot/
   │   ├── img1.jpg
   │   └── img2.jpg
   └── ...
   ```

2. **Train a model**:
   ```bash
   python train_disease_classifier.py --dataset_path ./my_disease_dataset --epochs 25
   ```

3. **Test the model**:
   ```bash
   python predict_disease.py --model_path ./models/disease_classifier_*.pth --image_path ./test.jpg
   ```

## Integration with Existing App

See `integration_instructions.md` for detailed instructions on:
- Dataset preparation guidelines
- Training best practices
- Integration with the existing MLProcessor
- Performance optimization tips
- Troubleshooting common issues

## Files Generated

### Training Output
- `disease_classifier_YYYYMMDD_HHMMSS.pth` - Trained model file
- `class_mapping_YYYYMMDD_HHMMSS.json` - Class names and indices  
- `training_history_YYYYMMDD_HHMMSS.png` - Training progress plot

### Prediction Output
- Console output with disease predictions and confidence
- Optional JSON output for programmatic use
- Optional results saved to JSON file

## Requirements

**Core dependencies** (from requirements.txt):
- PyTorch (`torch>=2.0.0`)
- torchvision (`torchvision>=0.15.0`) 
- Pillow for image processing
- matplotlib for training visualizations

**Installation**:
```bash
pip install -r requirements.txt
```

## Testing

Run the test script to validate everything works:
```bash
python test_pipeline.py
```

This validates:
- Script structure and syntax
- Dataset organization logic  
- Integration instructions completeness
- Argument parsing setup

## Key Advantages Over ImageNet Matching

1. **Clinical Accuracy**: Models trained on actual disease images vs generic ImageNet classes
2. **Domain Specificity**: Separate models for plants, humans, animals
3. **Confidence Scores**: Reliable probability estimates for predictions
4. **Customizable**: Add new diseases by retraining with updated datasets
5. **Performance**: Faster inference and higher accuracy on disease-specific tasks

## Example Workflow

```bash
# 1. Organize disease images by class
mkdir plant_diseases
mkdir plant_diseases/tomato_blight
mkdir plant_diseases/rose_black_spot
# ... add images to respective folders

# 2. Train the classifier
python train_disease_classifier.py \
    --dataset_path ./plant_diseases \
    --output_dir ./models \
    --epochs 30 \
    --batch_size 16

# 3. Test on new images
python predict_disease.py \
    --model_path ./models/disease_classifier_20240101_120000.pth \
    --image_path ./new_disease_photo.jpg \
    --show_all

# 4. Integrate with existing app (see integration_instructions.md)
```

This transforms the application from generic ImageNet classification to specialized disease detection, making it significantly more accurate and clinically relevant.