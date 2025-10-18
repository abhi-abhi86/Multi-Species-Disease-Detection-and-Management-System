# 🎯 Algorithms Quick Reference

Quick reference guide for algorithms used in the Multi-Species Disease Detection System.

> 📖 For detailed documentation, see [ALGORITHMS.md](ALGORITHMS.md)

---

## At a Glance

| Algorithm | Purpose | Accuracy/Performance |
|-----------|---------|---------------------|
| **MobileNetV2** | Image classification | 92-97% accuracy |
| **Transfer Learning** | Efficient training | Trains in minutes vs hours |
| **Fuzzy Matching** | Symptom matching | 60-100% similarity scores |
| **Adam Optimizer** | Weight updates | Faster convergence than SGD |
| **Data Augmentation** | Dataset expansion | 3x effective dataset size |

---

## Image Classification Pipeline

```
Input Image (Any Size)
    ↓
Resize to 224×224
    ↓
Normalize (ImageNet Stats)
    ↓
MobileNetV2 Feature Extraction
    ↓
Fully Connected Layer
    ↓
Softmax Activation
    ↓
Disease Class + Confidence %
```

**Key Parameters:**
- Input: 224×224 RGB image
- Output: Probability distribution over N disease classes
- Threshold: 50% confidence minimum
- Inference Time: ~2-5 seconds per image

---

## Symptom Matching Process

```
User Symptoms (Text)
    ↓
Lowercase & Clean
    ↓
Fuzzy Match (Levenshtein Distance)
    ↓
Top 3 Matches with Scores
    ↓
Filter by Threshold (60%+)
    ↓
Best Match Disease
```

**Key Parameters:**
- Strong Match: ≥75% similarity
- Weak Match: 60-74% similarity
- No Match: <60% similarity

---

## Training Process

```
Dataset Preparation
    ↓
Load MobileNetV2 (ImageNet Weights)
    ↓
Freeze Convolutional Layers
    ↓
Replace Final Classifier
    ↓
Data Augmentation Applied
    ↓
Train with Adam Optimizer
    ↓
Monitor Cross-Entropy Loss
    ↓
Save Best Model
```

**Key Hyperparameters:**
- Epochs: 20
- Batch Size: 16
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Cross-Entropy

---

## Mathematical Formulas

### Softmax (Classification)
```
P(class_i) = exp(z_i) / Σ(exp(z_j)) for all j
```

### Cross-Entropy Loss
```
L = -Σ y_i × log(ŷ_i)
```

### Adam Optimizer
```
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

### Levenshtein Distance
```
Minimum edits (insert/delete/substitute) to transform string A to string B
```

---

## Code Locations

| Algorithm | File | Lines |
|-----------|------|-------|
| MobileNetV2 Setup | `ml_processor.py` | 86-91 |
| Image Prediction | `ml_processor.py` | 111-253 |
| Symptom Matching | `ml_processor.py` | 255-282 |
| Transfer Learning | `train_disease_classifier.py` | 164-180 |
| Data Augmentation | `train_disease_classifier.py` | 136-148 |
| Adam Optimizer | `train_disease_classifier.py` | 183-191 |

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Model Size | 14 MB |
| Parameters | 3.5M (trainable: ~128K) |
| Training Time | 5-10 minutes (20 epochs) |
| Inference Time | 2-5 seconds per image |
| Accuracy (Test Set) | 92-97% |
| Minimum Confidence | 50% |

---

## Algorithm Selection Rationale

### Why MobileNetV2?
✅ **Efficiency** - Designed for mobile/edge devices  
✅ **Speed** - Fast inference time  
✅ **Accuracy** - Proven high accuracy  
✅ **Size** - Small model footprint  

### Why Transfer Learning?
✅ **Less Data Needed** - Works with smaller datasets  
✅ **Faster Training** - Minutes instead of hours  
✅ **Better Generalization** - Pre-learned features  
✅ **Less Overfitting** - Especially with limited data  

### Why Fuzzy Matching?
✅ **Typo Tolerance** - Handles spelling errors  
✅ **Flexible** - Works with partial matches  
✅ **Ranked Results** - Provides top N matches  
✅ **Medical Terms** - Handles complex terminology  

### Why Adam Optimizer?
✅ **Adaptive** - Different learning rates per parameter  
✅ **Fast Convergence** - Faster than SGD  
✅ **Robust** - Handles sparse gradients well  
✅ **Less Tuning** - Fewer hyperparameters to adjust  

---

## Future Enhancements

| Planned | Target | Expected Improvement |
|---------|--------|---------------------|
| Vision Transformer (ViT) | Q1 2026 | +3-5% accuracy |
| BERT for NLP | Q2 2026 | Better context understanding |
| Ensemble Models | Q2 2026 | +2-3% reliability |
| Active Learning | Q3 2026 | Continuous improvement |

---

## FAQs

**Q: Why not use a larger model like ResNet-50?**  
A: MobileNetV2 provides the best balance of speed, accuracy, and size for real-time applications on various devices.

**Q: Can I train with fewer than 20 epochs?**  
A: Yes, but accuracy may be lower. Monitor the validation loss to determine optimal stopping point.

**Q: What's the minimum dataset size needed?**  
A: At least 50 images per class recommended. Transfer learning helps with smaller datasets.

**Q: How accurate is the symptom matching?**  
A: Fuzzy matching achieves 60-100% similarity. Higher scores indicate better matches.

---

**Version:** 1.0  
**Last Updated:** 2025-10-18  
**See Also:** [ALGORITHMS.md](ALGORITHMS.md) for complete technical details
