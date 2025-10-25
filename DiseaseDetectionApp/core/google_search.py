
import os
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError





API_KEY = os.environ.get("GOOGLE_API_KEY")
CSE_ID = os.environ.get("GOOGLE_CSE_ID")

def _google_search_api_call(query, **kwargs):
    """A helper function to perform a Google Custom Search API call."""
    if not API_KEY or not CSE_ID:
        error_message = "API_KEY or CSE_ID environment variables not set."
        print(f"Error: {error_message}")
        return {"error": error_message}
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        result = service.cse().list(q=query, cx=CSE_ID, **kwargs).execute()
        return result
    except HttpError as e:
        print(f"An HTTP error occurred during Google search: {e}")
        return {"error": f"HTTP Error {e.resp.status}: {e.reason}"}
    except Exception as e:
        print(f"An unexpected error occurred during Google search: {e}")
        return {"error": str(e)}

def search_google_images(query):
    """
    Searches Google Images for the given query using the official API and returns the URL of the first image.
    """
    print(f"Performing API image search for: '{query}'")


    results = _google_search_api_call(query, searchType='image', num=1)

    if "error" in results:
        return None


    if 'items' in results and len(results['items']) > 0:
        return results['items'][0].get('link')

    print(f"No image results found for '{query}'.")
    return None

def search_google_for_summary(query):
    """
    Performs a Google search using the official API and returns the summary snippet of the first result.
    """
    print(f"Performing API summary search for: '{query}'")

    results = _google_search_api_call(query, num=1)

    if "error" in results:
        return f"Could not perform search. Please check API setup. Error: {results['error']}"


    if 'items' in results and len(results['items']) > 0:
        return results['items'][0].get('snippet')

    print(f"No text results found for '{query}'.")
    return "No summary could be found for this topic."
