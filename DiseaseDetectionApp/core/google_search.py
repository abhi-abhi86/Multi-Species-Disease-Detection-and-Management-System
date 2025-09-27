# DiseaseDetectionApp/core/google_search.py
import requests
from bs4 import BeautifulSoup

def search_google_images(query):
    """
    Searches Google Images for the given query and returns the URL of a relevant image.
    NOTE: This function relies on scraping Google's search results page, which is
    unstable and may break if Google changes its page layout. This version uses a more
    robust method than the original but is still subject to breaking.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image tags. We look for images that are likely to be search results.
        # Google often uses specific classes for its main image results.
        # Note: These class names ('rg_i', 'YQ4gaf') can change.
        image_tags = soup.find_all('img', class_=['rg_i', 'YQ4gaf'])

        for img in image_tags:
            # The actual URL is often in 'data-src' for lazy-loaded images,
            # but we also check 'src' as a fallback.
            src = img.get('data-src') or img.get('src')
            if src and src.startswith('http'):
                return src

        # If the specific class search fails, try a more general search for any image tag.
        print("Warning: Specific image class search failed. Using broader search.")
        all_images = soup.find_all("img")
        for img in all_images:
            src = img.get("src")
            if src and src.startswith("https://"):
                # Avoid small base64 encoded images or UI icons from gstatic
                if 'base64' not in src and 'gstatic' not in src:
                    return src

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image search results: {e}")
    except Exception as e:
        print(f"An error occurred during image scraping: {e}")

    return None


def search_google_for_summary(query):
    """
    Performs a Google search and attempts to find a summary snippet.
    NOTE: This is based on scraping and may break if Google changes its HTML structure.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for a "knowledge panel" or "featured snippet" which often has these classes.
        # This selector is more general and might be more resilient to changes.
        description_tag = soup.find("div", class_=["BNeawe", "s3v9rd", "AP7Wnd"])
        if description_tag:
            return description_tag.get_text()

    except requests.exceptions.RequestException as e:
        print(f"Error during Google search for summary: {e}")
    
    return None
