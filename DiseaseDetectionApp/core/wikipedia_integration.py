#wikipedia_integration.py
import wikipedia
from core.google_search import search_google_for_summary

def get_wikipedia_summary(disease_name):
    """
    Fetches a summary of the given disease from Wikipedia.
    If Wikipedia fails, it falls back to a Google search for a summary.
    """
    try:
        # Try Wikipedia first, auto_suggest=False to prevent incorrect matches
        summary = wikipedia.summary(disease_name, sentences=3, auto_suggest=False)
        return summary
    except wikipedia.exceptions.PageError:
        print(f"No Wikipedia page for '{disease_name}'. Falling back to Google search.")
        summary = search_google_for_summary(disease_name + " disease")
        return summary if summary else "No Wikipedia page found, and Google search returned no summary."
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            # Try the first option from disambiguation
            summary = wikipedia.summary(e.options[0], sentences=3)
            return summary
        except Exception:
            print(f"Could not resolve Wikipedia disambiguation for '{disease_name}'. Falling back to Google search.")
            summary = search_google_for_summary(disease_name + " disease")
            return summary if summary else "Could not resolve Wikipedia disambiguation, and Google search returned no summary."
    except Exception as e:
        print(f"An error occurred with Wikipedia: {e}. Falling back to Google search.")
        summary = search_google_for_summary(disease_name + " disease")
        return summary if summary else f"An error occurred while fetching from Wikipedia, and Google search also failed."
