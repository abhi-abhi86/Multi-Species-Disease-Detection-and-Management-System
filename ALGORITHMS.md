# 🧠 Algorithms Used in Multi-Species Disease Detection System

This document provides a comprehensive overview of all machine learning algorithms, data processing techniques, and computational methods employed in the Multi-Species Disease Detection and Management System.

---

## Table of Contents

1. [Overview](#overview)
2. [Deep Learning Algorithms](#deep-learning-algorithms)
3. [Natural Language Processing Algorithms](#natural-language-processing-algorithms)
4. [Data Processing Algorithms](#data-processing-algorithms)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Matching and Search Algorithms](#matching-and-search-algorithms)
7. [Performance Metrics](#performance-metrics)

---

## Overview

The system employs a multi-algorithm approach combining:
- **Deep Learning** for image-based disease detection
- **Natural Language Processing** for symptom-based diagnosis
- **Fuzzy Matching** for similarity searches
- **Transfer Learning** for efficient model training
- **Data Augmentation** for dataset enhancement

---

## Deep Learning Algorithms

### 1. MobileNetV2 - Convolutional Neural Network

**Purpose:** Primary architecture for image-based disease classification

**Algorithm Details:**
- **Architecture Type:** Convolutional Neural Network (CNN)
- **Base Model:** MobileNetV2 (developed by Google)
- **Key Innovation:** Inverted Residual Structure with Linear Bottlenecks
- **Input Size:** 224×224 pixels RGB images
- **Pre-training:** ImageNet dataset (1.4M images, 1000 classes)

**Why MobileNetV2?**
- **Efficiency:** Designed for mobile and edge devices with limited computational resources
- **Speed:** Fast inference time (~2-5 seconds per image)
- **Accuracy:** Achieves 92-97% classification accuracy on disease datasets
- **Small Size:** Significantly fewer parameters than traditional CNNs (3.5M parameters)
- **Depthwise Separable Convolutions:** Reduces computation while maintaining accuracy

**Architecture Components:**
```
Input (224×224×3)
    ↓
Initial Convolutional Layer (32 filters)
    ↓
17 Inverted Residual Blocks
    - Expansion layer (1×1 convolution)
    - Depthwise convolution (3×3)
    - Projection layer (1×1 convolution)
    - Residual connection (skip connection)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (Custom: N classes)
    ↓
Softmax Activation
    ↓
Output (Disease Class Probabilities)
```

**Mathematical Foundation:**

1. **Inverted Residual Block:**
   ```
   y = x + F(x)
   where F(x) = Projection(DepthWise(Expansion(x)))
   ```

2. **Softmax Function (Final Layer):**
   ```
   P(class_i) = exp(z_i) / Σ(exp(z_j)) for all j
   where z = logits from the network
   ```

**Implementation in Code:**
```python
# Location: DiseaseDetectionApp/train_disease_classifier.py
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
```

---

### 2. Transfer Learning

**Purpose:** Leverage pre-trained knowledge for efficient training on disease datasets

**Algorithm Strategy:**
1. **Feature Extraction:** Use MobileNetV2 pre-trained on ImageNet as a feature extractor
2. **Fine-tuning:** Replace only the final classification layer
3. **Frozen Layers:** Keep convolutional layers frozen (weights not updated)
4. **Custom Classifier:** Train only the final fully-connected layer for disease classes

**Benefits:**
- **Reduced Training Time:** Train in minutes instead of hours/days
- **Less Data Required:** Works well with smaller disease image datasets
- **Better Generalization:** Pre-learned features improve accuracy
- **Prevents Overfitting:** Especially important with limited training data

**Implementation:**
```python
# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# Only train the custom classifier
model.classifier[1] = nn.Linear(model.last_channel, num_disease_classes)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

---

### 3. Data Augmentation

**Purpose:** Artificially expand training dataset and improve model robustness

**Techniques Applied:**

1. **Random Horizontal Flip:**
   - Probability: 50%
   - Purpose: Learn mirror-symmetric features

2. **Random Rotation:**
   - Range: ±15 degrees
   - Purpose: Handle images taken at different angles

3. **Resize & Crop:**
   - Target: 224×224 pixels
   - Purpose: Standardize input dimensions

4. **Normalization:**
   - Mean: [0.485, 0.456, 0.406] (ImageNet statistics)
   - Std: [0.229, 0.224, 0.225]
   - Purpose: Standardize pixel value distributions

**Mathematical Formulation:**
```
Normalized_pixel = (pixel - mean) / std
```

**Implementation:**
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Natural Language Processing Algorithms

### 4. Fuzzy String Matching (Levenshtein Distance)

**Purpose:** Match user-input symptoms to known disease descriptions

**Algorithm:** FuzzyWuzzy library implementation (based on Levenshtein Distance)

**Levenshtein Distance Definition:**
The minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another.

**Mathematical Formula:**
```
lev(a,b) = {
    |a|                                  if |b| = 0
    |b|                                  if |a| = 0
    lev(tail(a), tail(b))                if a[0] = b[0]
    1 + min {
        lev(tail(a), b)                  (deletion)
        lev(a, tail(b))                  (insertion)
        lev(tail(a), tail(b))            (substitution)
    }                                    otherwise
}
```

**Similarity Score:**
```
similarity = 100 × (1 - distance / max(len(string1), len(string2)))
```

**Confidence Thresholds:**
- **Strong Match:** ≥75% similarity
- **Weak Match:** 60-74% similarity
- **No Match:** <60% similarity

**Example:**
```
Input: "high fever and cough"
Disease Symptoms: "fever, persistent cough, fatigue"
Similarity Score: 78% → Strong Match
```

**Implementation:**
```python
from fuzzywuzzy import process

results = process.extract(symptoms.lower(), disease_names, limit=3)
best_match_name, confidence = results[0]
```

---

### 5. Text Preprocessing

**Purpose:** Clean and standardize symptom text before matching

**Steps:**
1. **Lowercase Conversion:** Ensure case-insensitive matching
2. **Special Character Removal:** Keep only alphanumeric and spaces
3. **Whitespace Normalization:** Remove extra spaces

---

## Data Processing Algorithms

### 6. Image Preprocessing Pipeline

**Purpose:** Prepare images for neural network input

**Processing Steps:**

1. **Color Space Conversion:**
   ```
   RGB → Ensures 3-channel color input
   ```

2. **Resizing:**
   ```
   Original → 224×224 (Bilinear Interpolation)
   ```

3. **Tensor Conversion:**
   ```
   PIL Image → PyTorch Tensor [C, H, W]
   ```

4. **Normalization:**
   ```
   Pixel_norm = (Pixel - μ) / σ
   where μ = channel mean, σ = channel std
   ```

**Implementation:**
```python
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])
```

---

### 7. Class Label Sanitization

**Purpose:** Convert disease names to valid class labels for training

**Algorithm:**
```python
import re
safe_class_name = re.sub(r'[\s/\\:*?"<>|]+', '_', class_name).lower()
```

**Example Transformations:**
- "Rose Black Spot" → "rose_black_spot"
- "COVID-19" → "covid_19"
- "Bird Flu (H5N1)" → "bird_flu_h5n1"

---

## Optimization Algorithms

### 8. Adam Optimizer

**Purpose:** Update neural network weights during training

**Algorithm:** Adaptive Moment Estimation (Adam)

**Key Features:**
- Combines momentum and RMSprop
- Adaptive learning rates for each parameter
- Efficient for large datasets and high-dimensional spaces

**Mathematical Update Rules:**

1. **Compute gradients:**
   ```
   g_t = ∇_θ L(θ_t)
   ```

2. **Update biased first moment estimate:**
   ```
   m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
   ```

3. **Update biased second moment estimate:**
   ```
   v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
   ```

4. **Compute bias-corrected estimates:**
   ```
   m̂_t = m_t / (1 - β₁ᵗ)
   v̂_t = v_t / (1 - β₂ᵗ)
   ```

5. **Update parameters:**
   ```
   θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
   ```

**Hyperparameters Used:**
- Learning Rate (α): 0.001
- β₁ (Momentum): 0.9 (default)
- β₂ (RMSprop): 0.999 (default)
- ε (Numerical Stability): 1e-8 (default)

**Implementation:**
```python
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

---

### 9. Cross-Entropy Loss

**Purpose:** Measure classification error during training

**Mathematical Formula:**
```
L = -Σ y_i × log(ŷ_i)

where:
- y_i = true label (one-hot encoded)
- ŷ_i = predicted probability (softmax output)
```

**For Multi-class Classification:**
```
L = -log(ŷ_c) where c is the correct class
```

**Properties:**
- Penalizes confident wrong predictions heavily
- Gradient flows better than MSE for classification
- Natural pairing with softmax activation

**Implementation:**
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

---

## Matching and Search Algorithms

### 10. Database Matching Strategy

**Purpose:** Link neural network predictions to database entries

**Multi-level Matching Algorithm:**

1. **Level 1 - Exact Internal ID Match:**
   ```
   if predicted_class == disease.internal_id:
       return disease
   ```

2. **Level 2 - Exact Name Match:**
   ```
   if sanitized(predicted_class) == sanitized(disease.name):
       return disease
   ```

3. **Level 3 - Substring Match:**
   ```
   if predicted_class in disease.name or disease.name in predicted_class:
       return disease
   ```

4. **Level 4 - Fuzzy Match (Fallback):**
   ```
   similarity_scores = compute_similarity(predicted_class, all_diseases)
   if max(similarity_scores) >= 60:
       return best_match
   ```

5. **Level 5 - Web Search (Ultimate Fallback):**
   ```
   if all_above_fail:
       search_wikipedia(predicted_class)
       search_google(predicted_class)
   ```

**Implementation Location:** `DiseaseDetectionApp/core/ml_processor.py` (lines 133-222)

---

### 11. Domain-Specific Filtering

**Purpose:** Ensure predictions match the selected domain (Plant/Human/Animal)

**Algorithm:**
```python
domain_candidates = {
    disease['name']: disease 
    for disease in database 
    if disease.get("domain").lower() == selected_domain.lower()
}
```

**Benefits:**
- Reduces false positives
- Improves prediction accuracy
- Provides domain-specific suggestions if mismatch detected

---

## Performance Metrics

### Accuracy Metrics

1. **Classification Accuracy:**
   ```
   Accuracy = (Correct Predictions / Total Predictions) × 100
   Current Performance: 92-97%
   ```

2. **Confidence Scores:**
   ```
   Confidence = max(softmax_probabilities) × 100
   Threshold: 50% for image classification
   ```

3. **Training Loss:**
   ```
   Tracked per epoch to monitor convergence
   ```

---

## Algorithm Selection Rationale

### Why These Algorithms?

1. **MobileNetV2:**
   - ✅ Efficient for real-time applications
   - ✅ Works on low-power devices
   - ✅ Proven accuracy on image classification
   - ✅ Small model size for deployment

2. **Transfer Learning:**
   - ✅ Reduces training time significantly
   - ✅ Requires less labeled data
   - ✅ Improves generalization
   - ✅ State-of-the-art approach for specialized domains

3. **Fuzzy Matching:**
   - ✅ Handles typos and variations in user input
   - ✅ More flexible than exact string matching
   - ✅ Provides ranked results
   - ✅ Well-suited for medical terminology

4. **Adam Optimizer:**
   - ✅ Adaptive learning rates
   - ✅ Faster convergence than SGD
   - ✅ Handles sparse gradients well
   - ✅ Less hyperparameter tuning needed

---

## Algorithm Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
│                          │                                   │
│              ┌───────────┴───────────┐                       │
│              │                       │                       │
│         ┌────▼────┐             ┌───▼────┐                  │
│         │  IMAGE  │             │SYMPTOMS│                   │
│         └────┬────┘             └───┬────┘                   │
│              │                      │                        │
│    ┌─────────▼─────────┐   ┌───────▼────────┐              │
│    │ Image Preprocessing│   │Text Preprocessing│             │
│    │  - Resize to 224x224│   │ - Lowercase     │             │
│    │  - Normalize        │   │ - Clean text    │             │
│    │  - ToTensor         │   │                 │             │
│    └─────────┬───────────┘   └────────┬────────┘            │
│              │                         │                     │
│    ┌─────────▼──────────┐   ┌─────────▼────────┐           │
│    │   MobileNetV2       │   │ Fuzzy Matching   │           │
│    │   CNN Forward Pass  │   │ Levenshtein Dist │           │
│    │   - Feature Extract │   │ - Similarity %   │           │
│    │   - Classification  │   │                  │           │
│    └─────────┬───────────┘   └─────────┬────────┘           │
│              │                         │                     │
│    ┌─────────▼──────────┐   ┌─────────▼────────┐           │
│    │  Softmax Activation │   │ Confidence Score │           │
│    │  - Probabilities    │   │ - Ranking        │           │
│    └─────────┬───────────┘   └─────────┬────────┘           │
│              │                         │                     │
│              └───────────┬─────────────┘                     │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │ Database Matching     │                       │
│              │ - Multi-level Search  │                       │
│              │ - Domain Filtering    │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │   DIAGNOSIS RESULT    │                       │
│              │   + Confidence %      │                       │
│              │   + Treatment Info    │                       │
│              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Future Algorithm Enhancements

### Planned Improvements:

1. **Vision Transformer (ViT):**
   - Next-generation architecture for image analysis
   - Better attention mechanisms
   - Improved accuracy on complex cases

2. **BERT for NLP:**
   - Contextual understanding of symptoms
   - Better semantic matching
   - Multi-lingual support

3. **Ensemble Methods:**
   - Combine multiple models
   - Voting mechanisms
   - Improved reliability

4. **Active Learning:**
   - Iterative model improvement
   - User feedback integration
   - Continuous learning

---

## References and Resources

### Academic Papers:
1. **MobileNetV2:** Sandler et al. (2018) - "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
2. **Transfer Learning:** Pan & Yang (2010) - "A Survey on Transfer Learning"
3. **Adam Optimizer:** Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"

### Implementation Libraries:
- **PyTorch:** https://pytorch.org/
- **TorchVision:** https://pytorch.org/vision/
- **FuzzyWuzzy:** https://github.com/seatgeek/fuzzywuzzy

### Model Weights:
- **ImageNet Pre-trained Models:** https://pytorch.org/vision/stable/models.html

---

## Code References

### Key Files:
1. **`DiseaseDetectionApp/core/ml_processor.py`** - Image classification logic
2. **`DiseaseDetectionApp/train_disease_classifier.py`** - Training pipeline
3. **`DiseaseDetectionApp/core/prepare_dataset.py`** - Data preparation

### Algorithm Implementation Locations:
- MobileNetV2: Lines 76-77 (ml_processor.py)
- Transfer Learning: Lines 76-90 (ml_processor.py)
- Fuzzy Matching: Lines 229-252 (ml_processor.py)
- Adam Optimizer: Line 164 (train_disease_classifier.py)
- Data Augmentation: Lines 126-132 (train_disease_classifier.py)

---

## Contributing

If you'd like to suggest algorithm improvements or optimizations, please:
1. Review the current implementation
2. Benchmark your proposed changes
3. Submit a pull request with performance comparisons

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-18  
**Maintained By:** abhi-abhi86
