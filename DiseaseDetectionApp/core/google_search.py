# DiseaseDetectionApp/core/google_search.py
import requests
import json
from bs4 import BeautifulSoup

def search_google_images(query):
    """
    Searches Google Images for the given query and returns the URL of a relevant image.
    NOTE: This function relies on scraping Google's search results page, which can
    change its structure at any time, potentially breaking this function.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google often embeds image data in a script tag as JSON. This is more reliable
        # than parsing img tags directly, which are often lazy-loaded.
        all_script_tags = soup.find_all("script")
        for script in all_script_tags:
            if script.string and 'AF_initDataCallback' in script.string:
                # Find the JSON-like object within the script tag
                for line in script.string.splitlines():
                    if 'ds:1' in line and 'id:sde_i' in line:
                        try:
                            # Extract the nested list that contains image data
                            json_str = line.split('return', 1)[1].strip().rstrip(';')
                            data = json.loads(json_str)
                            # Navigate through the complex structure to find the image URL
                            image_url = data[56][1][0][0][1][0][0][0]
                            if image_url.startswith('http'):
                                return image_url
                        except (json.JSONDecodeError, IndexError, TypeError) as e:
                            print(f"Could not parse image data from script tag: {e}")
                            continue # Try the next script tag

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image search results: {e}")
    
    # Fallback if the script parsing fails
    print("Warning: Could not use primary method for image scraping. Using fallback.")
    try:
        if 'soup' in locals():
            for img in soup.find_all("img"):
                if img.get("src") and img.get("src").startswith("https://"):
                    return img["src"]
    except Exception as e:
        print(f"Fallback image scraping failed: {e}")

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

