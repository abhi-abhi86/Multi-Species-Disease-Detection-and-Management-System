import random
import re
from core.wikipedia_integration import get_wikipedia_summary

def predict_from_image(image_path, domain, database):
    """
    Mocks an image-based prediction and fetches a Wikipedia summary.
    """
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if candidates:
        result = random.choice(candidates)
        confidence = random.uniform(0.75, 0.98)
        wiki_summary = get_wikipedia_summary(result['name'])
        return result, confidence, wiki_summary
    return None, 0, ""

def predict_from_symptoms(symptoms, domain, database):
    """
    Predicts from symptoms using keyword matching and fetches a Wikipedia summary.
    """
    candidates = [d for d in database if d.get("domain", "").lower() == domain.lower()]
    if not candidates:
        return None, 0, ""

    user_symptoms = set(re.findall(r'\b\w+\b', symptoms.lower()))
    if not user_symptoms:
        return None, 0, ""

    best_match = None
    max_score = 0

    for disease in candidates:
        disease_text = (
            disease.get('name', '') + ' ' +
            disease.get('description', '') + ' ' +
            ' '.join(disease.get('stages', {}).values())
        ).lower()
        disease_words = set(re.findall(r'\b\w+\b', disease_text))
        score = len(user_symptoms.intersection(disease_words))
        
        if score > max_score:
            max_score = score
            best_match = disease

    if best_match:
        confidence = min(1.0, max_score / 10.0)
        wiki_summary = get_wikipedia_summary(best_match['name'])
        return best_match, confidence, wiki_summary
    
    return None, 0, ""
