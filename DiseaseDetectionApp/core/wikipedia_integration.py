                                                   
import wikipedia
import requests

def get_wikipedia_summary(disease_name):
    """
    Fetches a concise summary of the given disease from Wikipedia.
    Includes robust error handling for common issues like disambiguation or page not found.
    Reduced sentences for faster loading.
    """
    try:
                                                                                                     
                                                                                                         
                                                          
        summary = wikipedia.summary(disease_name, sentences=2, auto_suggest=False, redirect=True)
        return summary
    except wikipedia.exceptions.PageError:
        print(f"No direct Wikipedia page found for '{disease_name}'.")
        return "No Wikipedia page was found for this specific topic."
    except wikipedia.exceptions.DisambiguationError as e:
                                                                                     
        try:
            first_option = e.options[0]
            print(f"'{disease_name}' was ambiguous. Trying first option: '{first_option}'")
                                                      
            summary = wikipedia.summary(first_option, sentences=2, auto_suggest=False)
            return summary
        except Exception as inner_e:
            print(f"Could not resolve Wikipedia disambiguation for '{disease_name}': {inner_e}")
            return "This term is ambiguous and a specific Wikipedia page could not be determined."
    except requests.exceptions.RequestException as e:
                                                                  
        print(f"Network error while fetching from Wikipedia: {e}")
        return "Could not connect to Wikipedia. Please check your internet connection."
    except Exception as e:
                                                                       
        print(f"An unexpected error occurred with the Wikipedia library: {e}")
        return "An error occurred while trying to fetch data from Wikipedia."
