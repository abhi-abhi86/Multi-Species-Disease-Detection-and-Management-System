# DiseaseDetectionApp/core/google_search.py
import requests
from bs4 import BeautifulSoup

def search_google_images(query):
    """
    Searches Google Images for the given query and returns the URL of a relevant image.
    This is a simple implementation and might be affected by changes to Google's search result page.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all image tags, skip the first one which is usually the Google logo
        image_tags = soup.find_all('img')[1:] 
        
        for img in image_tags:
            src = img.get('src')
            if src and src.startswith('http'):
                return src

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image search results: {e}")
    
    return None

def search_google_for_summary(query):
    """
    Performs a Google search and returns the text from the first result snippet.
    Note: This is based on scraping and may break if Google changes its HTML structure.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This is a common class for Google's description snippets, but it's fragile.
        description_tag = soup.find("div", class_="BNeawe s3v9rd AP7Wnd")
        if description_tag:
            return description_tag.get_text()

    except requests.exceptions.RequestException as e:
        print(f"Error during Google search for summary: {e}")
    
    return None
