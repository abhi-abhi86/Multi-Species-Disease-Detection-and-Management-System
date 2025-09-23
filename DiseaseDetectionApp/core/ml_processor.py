import random

def predict_from_image(image_path, domain, database):
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if candidates:
        return random.choice(candidates)
    return None

def predict_from_symptoms(symptoms, domain, database):
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if candidates:
        return random.choice(candidates)
    return None
