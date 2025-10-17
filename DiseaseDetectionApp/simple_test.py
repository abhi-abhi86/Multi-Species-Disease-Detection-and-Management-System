#!/usr/bin/env python3
"""
Simple test for Wikipedia image fetching without PyQt5 threading.
"""

import wikipedia
import requests

def get_wikipedia_image(disease_name):
    """
    Attempts to fetch the first available image URL from the Wikipedia page for the disease.
    Returns None if no images are found or an error occurs.
    """
    try:
        page = wikipedia.page(disease_name, auto_suggest=False, redirect=True)
        if page.images:
            # Return the first image URL that is likely a valid image (not SVG or icon)
            for img_url in page.images:
                if img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    return img_url
        return None
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError:
        return None
    except Exception:
        return None

def test_wikipedia_image(disease_name):
    print(f"Testing Wikipedia image for: {disease_name}")
    img_url = get_wikipedia_image(disease_name)
    if img_url:
        print(f"  Found image: {img_url}")
        # Try to fetch the image to verify it's accessible
        try:
            response = requests.head(img_url, timeout=5)
            if response.status_code == 200:
                print(f"  Image accessible (status: {response.status_code})")
            else:
                print(f"  Image not accessible (status: {response.status_code})")
        except Exception as e:
            print(f"  Error fetching image: {e}")
    else:
        print("  No image found on Wikipedia")
    print()

if __name__ == "__main__":
    test_wikipedia_image("Lumpy Skin Disease")
    test_wikipedia_image("Acne Vulgaris")
    test_wikipedia_image("Nonexistent Disease")
    test_wikipedia_image("Ringworm")
