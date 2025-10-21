# DiseaseDetectionApp/core/ncbi_integration.py
import requests
import xml.etree.ElementTree as ET

def get_pubmed_summary(disease_name, max_results=2):
    """
    Fetches recent research summaries for a given disease from the PubMed database.
    This version has improved error handling and clearer status messages.
    Reduced max_results to 2 for faster loading.
    """
    print(f"Searching PubMed for articles related to: '{disease_name}'")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    # Create a more specific search term to improve relevance.
    search_term = f'"{disease_name}"[Title/Abstract]'

    try:
        # Step 1: Search PubMed for article IDs (PMIDs).
        # Use a timeout to prevent the app from hanging on slow network requests.
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={max_results}&sort=relevance"
        response = requests.get(search_url, timeout=10)  # Reduced timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx).

        root = ET.fromstring(response.content)
        id_list = [elem.text for elem in root.findall('.//Id')]

        if not id_list:
            print("No relevant articles found on PubMed.")
            return "No recent research articles were found on PubMed for this topic."

        # Step 2: Fetch the summaries for the found article IDs.
        ids = ",".join(id_list)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml&rettype=abstract"
        fetch_response = requests.get(fetch_url, timeout=10)  # Reduced timeout
        fetch_response.raise_for_status()

        fetch_root = ET.fromstring(fetch_response.content)
        articles = fetch_root.findall('.//PubmedArticle')

        summary_texts = []
        for article in articles:
            title_elem = article.find('.//ArticleTitle')
            abstract_elem = article.find('.//AbstractText')

            # Ensure elements exist before accessing their text attribute.
            title = title_elem.text if title_elem is not None and title_elem.text else "No Title Available"
            abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else "No Abstract Available"

            # Format the output nicely for display in the UI.
            # Reduced abstract length for faster display
            summary_texts.append(f"  • <b>{title}</b><br>    <i>{abstract[:200]}...</i>")

        if not summary_texts:
            return "Found articles, but could not extract their summaries."

        return "<br><br>".join(summary_texts)

    except requests.exceptions.Timeout:
        print("PubMed request timed out.")
        return "The request to the PubMed database timed out. Please try again later."
    except requests.exceptions.RequestException as e:
        print(f"A network error occurred while contacting PubMed: {e}")
        return "Could not connect to the PubMed database. Please check your internet connection."
    except ET.ParseError as e:
        print(f"Error parsing XML response from PubMed: {e}")
        return "Failed to parse the response from the PubMed database. The data may be malformed."
    except Exception as e:
        print(f"An unexpected error occurred during PubMed integration: {e}")
        return "An unexpected error occurred while fetching research data."
