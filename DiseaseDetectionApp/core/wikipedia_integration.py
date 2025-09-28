# DiseaseDetectionApp/core/wikipedia_integration.py
import wikipedia
import requests

def get_wikipedia_summary(disease_name):
    """
    Fetches a concise summary of the given disease from Wikipedia.
    Includes robust error handling for common issues like disambiguation or page not found.
    """
    try:
        # `auto_suggest=False` prevents Wikipedia from guessing a different page, which can be wrong.
        # `redirect=True` allows it to follow simple page redirects (e.g., "Mange" -> "Sarcoptic mange").
        summary = wikipedia.summary(disease_name, sentences=4, auto_suggest=False, redirect=True)
        return summary
    except wikipedia.exceptions.PageError:
        print(f"No direct Wikipedia page found for '{disease_name}'.")
        return "No Wikipedia page was found for this specific topic."
    except wikipedia.exceptions.DisambiguationError as e:
        # If the term is ambiguous (e.g., "Ringworm"), try the first option provided.
        try:
            first_option = e.options[0]
            print(f"'{disease_name}' was ambiguous. Trying first option: '{first_option}'")
            summary = wikipedia.summary(first_option, sentences=4, auto_suggest=False)
            return summary
        except Exception as inner_e:
            print(f"Could not resolve Wikipedia disambiguation for '{disease_name}': {inner_e}")
            return "This term is ambiguous and a specific Wikipedia page could not be determined."
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (no internet, DNS issues).
        print(f"Network error while fetching from Wikipedia: {e}")
        return "Could not connect to Wikipedia. Please check your internet connection."
    except Exception as e:
        # Catch any other unexpected errors from the wikipedia library.
        print(f"An unexpected error occurred with the Wikipedia library: {e}")
        return "An error occurred while trying to fetch data from Wikipedia."
