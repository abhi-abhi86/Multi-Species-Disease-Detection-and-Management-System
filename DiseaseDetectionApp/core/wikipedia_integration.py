#wikipedia_integration.py
import wikipedia

def get_wikipedia_summary(disease_name):
    """
    Fetches a summary of the given disease from Wikipedia.
    """
    try:
        # Set a shorter summary length to keep it concise
        summary = wikipedia.summary(disease_name, sentences=3)
        return summary
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this disease."
    except wikipedia.exceptions.DisambiguationError as e:
        # In case of multiple options, take the first one
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            return summary
        except Exception:
            return "Could not resolve Wikipedia disambiguation."
    except Exception as e:
        return f"An error occurred while fetching from Wikipedia: {e}"
