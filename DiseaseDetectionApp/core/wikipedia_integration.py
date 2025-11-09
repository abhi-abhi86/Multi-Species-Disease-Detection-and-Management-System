
import wikipedia
import requests

def get_wikipedia_summary(disease_name):
    """
    Fetches a concise summary of the given disease from Wikipedia.
    Includes robust error handling for common issues like disambiguation or page not found.
    Uses auto-suggest to find similar pages and includes a search fallback for better information retrieval.
    """
    try:
        # Enable auto_suggest to allow Wikipedia to suggest similar pages
        summary = wikipedia.summary(disease_name, sentences=3, auto_suggest=True, redirect=True)
        return summary
    except wikipedia.exceptions.PageError:
        print(f"No direct Wikipedia page found for '{disease_name}'. Attempting search fallback.")
        # Fallback: Search for the disease and try the first result
        try:
            search_results = wikipedia.search(disease_name, results=1)
            if search_results:
                first_result = search_results[0]
                print(f"Search fallback: Trying '{first_result}' for '{disease_name}'.")
                summary = wikipedia.summary(first_result, sentences=3, auto_suggest=True, redirect=True)
                return summary
            else:
                print(f"No search results found for '{disease_name}'.")
                return None
        except Exception as fallback_e:
            print(f"Search fallback failed for '{disease_name}': {fallback_e}")
            return None
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            first_option = e.options[0]
            print(f"'{disease_name}' was ambiguous. Trying first option: '{first_option}' with auto_suggest.")
            # Enable auto_suggest for disambiguation resolution
            summary = wikipedia.summary(first_option, sentences=3, auto_suggest=True)
            return summary
        except Exception as inner_e:
            print(f"Could not resolve Wikipedia disambiguation for '{disease_name}': {inner_e}")
            # Try search fallback for disambiguation
            try:
                search_results = wikipedia.search(disease_name, results=1)
                if search_results:
                    first_result = search_results[0]
                    print(f"Disambiguation search fallback: Trying '{first_result}' for '{disease_name}'.")
                    summary = wikipedia.summary(first_result, sentences=3, auto_suggest=True, redirect=True)
                    return summary
                else:
                    return None
            except Exception as search_e:
                print(f"Disambiguation search fallback failed for '{disease_name}': {search_e}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching from Wikipedia: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with the Wikipedia library: {e}")
        return None
