# Integration Instructions for Real Image-to-Disease Detection

## 1. Prepare your dataset

Organize your images by disease in folders like:
```
dataset/
  Disease1/
  Disease2/
  ...
```

## 2. Train the model

Run:
```
python train_disease_classifier.py
```
This will create `disease_model.pt` and `class_to_name.json`.

## 3. Predict with your model

Test with:
```
python predict_disease.py
```

## 4. Integrate into your application

- Move both `disease_model.pt` and `class_to_name.json` to a known location in your app.
- In your disease prediction pipeline, use the code from `predict_disease.py` (especially the `predict_image()` function).
- After you get `pred_class`, use it to look up disease details in your database.

Example:
```python
disease_name = predict_image(image_path)
disease_info = next((d for d in database if d.get("name", "").lower() == disease_name.lower()), None)
```

## 5. Tips

- You can retrain the model any time you add more images.
- Make sure the disease names in your database match the folder names for best results.

---
